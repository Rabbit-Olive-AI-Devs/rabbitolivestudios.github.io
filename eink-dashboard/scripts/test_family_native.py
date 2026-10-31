"""Offline-only native parity. All DBs, identifiers, names and shown data are synthetic."""
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest

S = Path(__file__).resolve().parent
# Explicit test-only output root. Never infer a production state directory from repo location.
E = Path(os.environ['FAMILY_TEST_ROOT']) if 'FAMILY_TEST_ROOT' in os.environ else Path(tempfile.mkdtemp(prefix='family-native-', dir=os.environ.get('TMPDIR')))
for subdir in ['evidence','cache/swift','tmp']:
    (E/subdir).mkdir(parents=True,exist_ok=True)
spec = importlib.util.spec_from_file_location('reference', S / 'family-pick.py')
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)
BINARY = E / 'evidence' / 'family-core-harness'


def run(o):
    return json.loads(subprocess.check_output([str(BINARY)], input=json.dumps(o).encode()))


class NativeParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert (S / 'family-core.swift').exists(), 'missing feature: in-process native family selection core'
        p = subprocess.run(['swiftc', '-module-cache-path', str(E / 'cache' / 'swift'),
                            str(S / 'family-core.swift'), str(S / 'family-core-harness.swift'),
                            '-lsqlite3', '-o', str(BINARY)], capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr

    def compare(self, cs, target='2026-10-19', shown=()):
        rnd, order = fp.choose(cs, list(shown), dt.date.fromisoformat(target))
        wire = [{**c, 'taken': c['taken'].isoformat()} for c in cs]
        got = run(dict(op='choose', date=target, candidates=wire, shown=list(shown)))
        expected = [{**c, 'taken': c['taken'].isoformat(), 'localIdentifier': c['uuid']+'/L0/001'} for c in order]
        self.assertEqual((got['round'], got['candidates']), (rnd, expected))
        return got

    def test_existing_seven_against_native(self):
        # Run the pick tests against a native choose bridge. Place tests do not use choose.
        spec = importlib.util.spec_from_file_location('existing', S / 'test_family_pick.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        original = mod.fp.choose
        def native(cs, shown, target):
            got = self.compare(cs, target.isoformat(), shown)
            return got['round'], [{**c, 'taken': dt.date.fromisoformat(c['taken'])} for c in got['candidates']]
        mod.fp.choose = native
        try:
            result = unittest.TestResult()
            unittest.defaultTestLoader.loadTestsFromModule(mod).run(result)
            self.assertEqual(result.testsRun, 11)
            self.assertEqual(result.errors + result.failures, [])
        finally:
            mod.fp.choose = original

    def test_age_round_stable_ties_leap_wrap_and_cap(self):
        target = dt.date(2026, 10, 19)
        c = lambda u,d,s='album',p=100: dict(uuid=u,taken=dt.date.fromisoformat(d),source=s,pixels=p)
        cases = [
          ([c('335',(target-dt.timedelta(days=335)).isoformat()), c('334',(target-dt.timedelta(days=334)).isoformat())], []),
          ([c('old','2020-10-19'), c('young','2026-10-18')], [dict(uuid='old',date='2025-01-01',round=4)]),
          ([c('a','2020-10-19'), c('b','2019-10-19')], [dict(uuid=u,date='2025-01-01',round=4) for u in ['a','b']]),
          ([c(str(i),'2020-10-19',p=100 if i<3 else i) for i in range(20)], []),
          ([c('a','2020-10-18'),c('b','2020-10-20'),c('f','2019-10-19','favourite',1)], []),
          ([c('young','2026-10-19'),c('young','2018-10-19','favourite')], []),
        ]
        for cs,shown in cases:
            with self.subTest(cs=cs): self.compare(cs, shown=shown)
        for date in ['2026-02-28','2028-02-29','2027-01-01','2026-12-31']:
            self.compare([c('leap','2020-02-29'),c('jan','2019-01-01'),c('dec','2019-12-31')],date)

    def test_sql_filters_offsets_names_and_duplicates(self):
        with tempfile.TemporaryDirectory(dir=E / 'tmp') as tmp:
            dbpath = Path(tmp) / 'synthetic.sqlite'
            db = sqlite3.connect(dbpath)
            db.executescript('''
              CREATE TABLE ZSHARE(Z_PK INTEGER PRIMARY KEY,ZTITLE TEXT);
              CREATE TABLE ZASSET(Z_PK INTEGER PRIMARY KEY,ZUUID TEXT,ZDATECREATED REAL,ZWIDTH INTEGER,ZHEIGHT INTEGER,ZKIND INTEGER DEFAULT 0,ZTRASHEDSTATE INTEGER DEFAULT 0,ZHIDDEN INTEGER DEFAULT 0,ZKINDSUBTYPE INTEGER,ZCOLLECTIONSHARE INTEGER,ZFAVORITE INTEGER DEFAULT 0,ZLATITUDE REAL,ZLONGITUDE REAL);
              CREATE TABLE ZADDITIONALASSETATTRIBUTES(ZASSET INTEGER,ZTIMEZONEOFFSET REAL);
              CREATE TABLE ZPERSON(Z_PK INTEGER PRIMARY KEY,ZFULLNAME TEXT,ZDISPLAYNAME TEXT);
              CREATE TABLE ZDETECTEDFACE(ZASSETFORFACE INTEGER,ZHIDDEN INTEGER,ZPERSONFORFACE INTEGER);
              INSERT INTO ZSHARE VALUES(5,'Family Album');
              INSERT INTO ZSHARE VALUES(6,'Different Album');
              INSERT INTO ZPERSON VALUES(1,'Synthetic Family','Alias');
              INSERT INTO ZPERSON VALUES(2,'Other Person','Synthetic Family');
              INSERT INTO ZPERSON VALUES(3,'Unrecognized','Other');
            ''')
            secs = (dt.datetime(2020,10,19,0,30)-fp.APPLE_EPOCH).total_seconds()
            for i in range(1,23):
                db.execute('INSERT INTO ZASSET(Z_PK,ZUUID,ZDATECREATED,ZWIDTH,ZHEIGHT,ZCOLLECTIONSHARE,ZFAVORITE) VALUES(?,?,?,?,?,?,?)',
                           (i,f'synthetic-{i}',secs,1600+i,1000,5 if i<14 else 6,1 if i>=12 else 0))
                db.execute('INSERT INTO ZADDITIONALASSETATTRIBUTES VALUES(?,?)',(i,-3600 if i==1 else (86400 if i==2 else None)))
                db.execute('INSERT INTO ZDETECTEDFACE VALUES(?,?,?)',(i,1 if i==15 else 0,3 if i==16 else (2 if i==17 else 1)))
            for i,col,val in [(3,'ZKIND',1),(4,'ZTRASHEDSTATE',1),(5,'ZHIDDEN',1),(6,'ZKINDSUBTYPE',10),(7,'ZWIDTH',1000),(8,'ZWIDTH',900),(9,'ZKINDSUBTYPE',0)]:
                db.execute(f'UPDATE ZASSET SET {col}=? WHERE Z_PK=?',(val,i))
            db.execute('UPDATE ZASSET SET ZLATITUDE=?, ZLONGITUDE=? WHERE Z_PK=1', (1.25, 2.5))
            db.commit()
            people={'synthetic':' Synthetic Family '}
            for target in ['2026-10-19','2026-10-18','2026-10-20']:
                candidates,found = fp.read_candidates(db,people)
                rnd,order=fp.choose(candidates,[],dt.date.fromisoformat(target))
                got=run(dict(op='sql',db=str(dbpath),date=target,people=people,shown=[]))
                self.assertEqual(got['albumFound'],found)
                self.assertEqual(got['readCandidates'],[{**c,'taken':c['taken'].isoformat()} for c in candidates])
                self.assertEqual(got['round'],rnd)
                self.assertEqual(got['candidates'],[{**c,'taken':c['taken'].isoformat(),'localIdentifier':c['uuid']+'/L0/001'} for c in order])
            # Python str.strip includes ASCII record/group separators as whitespace.
            people={'synthetic':'\x1cSynthetic Family\x1f'}
            expected,_=fp.read_candidates(db,people)
            got=run(dict(op='sql',db=str(dbpath),date='2026-10-19',people=people,shown=[]))
            self.assertEqual(got['readCandidates'],[{**c,'taken':c['taken'].isoformat()} for c in expected])
            # A private configured title must be bound as SQL data, not source text.
            alternate = "Synthetic Album's ? Title"
            db.execute('UPDATE ZSHARE SET ZTITLE=? WHERE Z_PK=5', (alternate,)); db.commit()
            configured = run(dict(op='sql',db=str(dbpath),date='2026-10-19',people=people,shown=[],albumTitle=alternate))
            self.assertTrue(configured['albumFound'])
            self.assertEqual(configured['readCandidates'],[{**c,'taken':c['taken'].isoformat()} for c in expected])
            default = run(dict(op='sql',db=str(dbpath),date='2026-10-19',people={},shown=[]))
            self.assertFalse(default['albumFound'])
            self.assertEqual(default['candidates'],[])
            db.execute('DELETE FROM ZSHARE WHERE Z_PK=5'); db.commit()
            got=run(dict(op='sql',db=str(dbpath),date='2026-10-19',people={},shown=[]))
            self.assertFalse(got['albumFound']); self.assertEqual(got['candidates'],[])
            db.close()

    def test_private_snapshot_cleanup_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory(dir=E / 'tmp') as tmp:
            root=Path(tmp); lib=root/'synthetic.photoslibrary'; (lib/'database').mkdir(parents=True)
            cache=root/'cache'; dbfile=lib/'database'/'Photos.sqlite'
            db=sqlite3.connect(dbfile)
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''CREATE TABLE ZSHARE(Z_PK INTEGER,ZTITLE TEXT);
              CREATE TABLE ZASSET(ZUUID TEXT,ZDATECREATED REAL,ZWIDTH INTEGER,ZHEIGHT INTEGER,ZKIND INTEGER,ZTRASHEDSTATE INTEGER,ZHIDDEN INTEGER,ZKINDSUBTYPE INTEGER,ZCOLLECTIONSHARE INTEGER,Z_PK INTEGER,ZFAVORITE INTEGER);
              CREATE TABLE ZADDITIONALASSETATTRIBUTES(ZASSET INTEGER,ZTIMEZONEOFFSET INTEGER);
              CREATE TABLE ZPERSON(Z_PK INTEGER,ZFULLNAME TEXT,ZDISPLAYNAME TEXT);
              INSERT INTO ZSHARE VALUES(1,'Family Album');
              INSERT INTO ZASSET VALUES('synthetic',0,1600,1000,0,0,0,0,1,1,0);'''); db.commit()
            (lib/'database'/'private-extra').write_text('never copy')
            before={p.name:p.read_bytes() for p in (lib/'database').iterdir()}
            got=run(dict(op='snapshot',library=str(lib),cache=str(cache)))
            self.assertEqual(got.get('count'),1)
            self.assertEqual(set(got['files']),{'Photos.sqlite','Photos.sqlite-wal','Photos.sqlite-shm'})
            self.assertEqual(list(cache.iterdir()),[])
            self.assertEqual(before,{p.name:p.read_bytes() for p in (lib/'database').iterdir()})
            self.assertEqual(run(dict(op='snapshot',library=str(lib),cache=str(cache),fail=True)),{'blocked':True})
            self.assertEqual(list(cache.iterdir()),[])
            alias=root/'alias';alias.symlink_to(lib,target_is_directory=True)
            self.assertEqual(run(dict(op='snapshot',library=str(alias),cache=str(cache))),{'blocked':True})
            db.close()
            dbfile.rename(lib/'database'/'original')
            dbfile.symlink_to(lib/'database'/'original')
            self.assertEqual(run(dict(op='snapshot',library=str(lib),cache=str(cache))),{'blocked':True})
            self.assertEqual(list(cache.iterdir()),[])
            # Full authorization is not evidence a DB is actually readable/queryable.
            dbfile.unlink(); dbfile.write_bytes(b'not a SQLite database')
            self.assertEqual(run(dict(op='snapshot',library=str(lib),cache=str(cache))),{'blocked':True})
            self.assertEqual(list(cache.iterdir()),[])

    def test_atomic_winner_only_publication_and_existing_request_rejection(self):
        with tempfile.TemporaryDirectory(dir=E / 'tmp') as tmp:
            root=Path(tmp)/'output'
            request='11111111-1111-4111-8111-111111111111'
            receipt=dict(request=request,date='2026-10-19',uuid='22222222-2222-4222-8222-222222222222',taken='2020-10-19',source='album',round=2,width=1600,height=1000)
            self.assertEqual(run(dict(op='publish',root=str(root),receipt=receipt)),{'ok':True})
            folder=root/request
            self.assertEqual(set(p.name for p in folder.iterdir()),{'photo.jpg','selection.json','completed.json'})
            self.assertEqual(json.loads((folder/'selection.json').read_text()),receipt)
            self.assertEqual(json.loads((folder/'completed.json').read_text()),dict(request=request,date='2026-10-19'))
            before={p.name:p.read_bytes() for p in folder.iterdir()}
            self.assertEqual(run(dict(op='publish',root=str(root),receipt=receipt)),{'blocked':True})
            self.assertEqual(before,{p.name:p.read_bytes() for p in folder.iterdir()})
            self.assertEqual(run(dict(op='publish',root=str(root),receipt={**receipt,'request':'../escape'})),{'blocked':True})

    def test_protected_denials_are_distinguished_from_candidate_download_failure(self):
        for domain,code,denied in [('PHPhotosErrorDomain',3310,True),('PHPhotosErrorDomain',3311,True),('NSCocoaErrorDomain',257,True),('NSCocoaErrorDomain',513,True),('NSPOSIXErrorDomain',1,True),('NSPOSIXErrorDomain',13,True),('PHPhotosErrorDomain',3169,False),('PHPhotosErrorDomain',3201,False)]:
            self.assertEqual(run(dict(op='denial',domain=domain,code=code))['denied'],denied)

    def test_download_fallback_width_threshold_and_denial_stop(self):
        cs=[dict(uuid=f'synthetic-{i}',taken='2020-10-19',source='album',pixels=1) for i in range(5)]
        got=run(dict(op='download',candidates=cs,widths=[-1,1023,1024,1600,1600]))
        self.assertEqual(got,dict(attempts=['synthetic-0','synthetic-1','synthetic-2'],winner='synthetic-2'))
        self.assertEqual(run(dict(op='download',candidates=cs,widths=[-2,1600,1600,1600,1600])),dict(attempts=['synthetic-0'],blocked=True))
        self.assertEqual(run(dict(op='download',candidates=cs,widths=[-1]*5)),dict(attempts=[c['uuid'] for c in cs],winner='none'))

    def test_authorization_precedes_any_metadata_read(self):
        # false represents limited, denied, restricted and notDetermined alike.
        self.assertEqual(run(dict(op='guard',authorized=False)),dict(reads=0,blocked=True))
        self.assertEqual(run(dict(op='guard',authorized=True)),dict(reads=1,blocked=False))

    def test_canonical_request_date(self):
        request='11111111-1111-4111-8111-111111111111'
        for date,req,valid in [('1500-02-29',request,False),('1582-10-10',request,True),('2026-10-19',request,True),('2028-02-29',request,True),('2026-02-29',request,False),('2026-1-19',request,False),('0000-01-01',request,False),('2026-10-19','../escape',False)]:
            self.assertEqual(run(dict(op='request',date=date,request=req))['valid'],valid)


if __name__ == '__main__': unittest.main(verbosity=2)
