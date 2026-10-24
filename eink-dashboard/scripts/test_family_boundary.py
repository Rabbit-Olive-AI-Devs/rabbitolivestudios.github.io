"""Offline source-boundary checks. Never execute the PhotoKit application."""
from pathlib import Path
import unittest
import os
import tempfile
import subprocess
S=Path(__file__).resolve().parent

class Boundary(unittest.TestCase):
    def test_unattended_dispatch_and_full_access_guard(self):
        s=(S/'family-fetch.swift').read_text()
        self.assertIn('--select-and-fetch',s)
        self.assertIn('FamilyCore.authorized(PHPhotoLibrary.authorizationStatus(for: .readWrite) == .authorized)',s)
        self.assertIn('FamilyFiles.withSnapshot',s)
        self.assertIn('Bundle.main.url(forResource: "people", withExtension: "json")',s)
        self.assertNotIn('Process(',s)
        self.assertNotIn('--authorize-photos',s)  # legacy interactive flow is sufficient; no new prompt path
        self.assertIn('status == .authorized || status == .limited',s)
        self.assertIn('PHPhotoLibrary.requestAuthorization',s)

    def test_consumer_no_library_db_roster_or_python(self):
        s=(S/'series-mailbox.mjs').read_text()
        start=s.index('function familyPhoto(date)')
        end=s.index('/**',start)
        body=s[start:end]
        for token in ['Photos.sqlite','family-pick.py','PHOTOS_LIB','people.json','python3']:
            self.assertNotIn(token,body)
        self.assertIn('selectFamilyPhoto',body)
        self.assertNotIn('process.env.PHOTOS_LIBRARY',s)
        self.assertNotIn('process.env.FAMILY_TMP',s)

    def test_builder_rejects_symlink_parent_without_writing_bundle(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
            root=Path(tmp); actual=root/'actual';actual.mkdir()
            alias=root/'alias';alias.symlink_to(actual,target_is_directory=True)
            result=subprocess.run(['bash',str(S/'build-family-fetch.sh'),'--output',str(alias/'EinkFamilyFetch.app'),'--without-people'],capture_output=True,text=True)
            self.assertEqual(result.returncode,2,result.stdout+result.stderr)
            self.assertEqual(list(actual.iterdir()),[])

    def test_build_is_explicit_staging_only_no_ls_and_no_live_replacement(self):
        s=(S/'build-family-fetch.sh').read_text()
        self.assertIn('--output',s)
        self.assertIn('--without-people',s)
        self.assertIn('family-core.swift',s)
        self.assertNotIn('lsregister',s)
        self.assertNotIn('APP="$HOME/Applications',s)
        self.assertIn('com.eink.family-fetch',s)
        self.assertIn('CFBundleVersion</key><string>1</string>',s)

if __name__=='__main__':unittest.main(verbosity=2)
