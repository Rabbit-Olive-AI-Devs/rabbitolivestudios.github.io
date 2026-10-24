"""Actual native lifecycle/locks with disposable processes and synthetic files only."""
from pathlib import Path
import json, os, subprocess, tempfile, time, unittest
S=Path(__file__).resolve().parent
class Lifecycle(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.build=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']);cls.binary=Path(cls.build.name)/'lifecycle'
  subprocess.run(['swiftc','-module-cache-path',os.environ['SWIFT_MODULECACHE_PATH'],str(S/'family-core.swift'),str(S/'family-lifecycle-harness.swift'),'-lsqlite3','-o',str(cls.binary)],check=True,capture_output=True)
 @classmethod
 def tearDownClass(cls):cls.build.cleanup()
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']);self.root=Path(self.tmp.name);self.home=self.root/'home';self.active=self.home/'Library/Caches/com.eink.family-fetch-control/active';self.active.mkdir(parents=True)
  self.request='11111111-1111-4111-8111-111111111111';self.day='2026-10-19';self.processes=[]
  self.owner={'request':self.request,'date':self.day,'pid':os.getpid(),'deadline':time.time()*1000+100000,'tmp':str(self.root/'out')};self.save()
 def tearDown(self):
  (self.root/'gate').touch()
  for p in self.processes:
   try:p.wait(timeout=3)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  self.tmp.cleanup()
 def save(self):(self.active/'owner.json').write_text(json.dumps(self.owner))
 def start(self,request=None,day=None):
  p=subprocess.Popen([str(self.binary),str(self.home),day or self.day,request or self.request,str(self.root/'ready'),str(self.root/'gate'),str(self.root/'protected')]);self.processes.append(p);return p
 def wait(self,file):
  for _ in range(200):
   if file.exists():return
   time.sleep(.01)
  self.fail('synthetic producer did not acknowledge')
 def terminal(self):return json.loads((self.active/'stopped.json').read_text())
 def test_healthy_and_actual_exclusive_native_lock(self):
  first=self.start();self.wait(self.root/'ready');second=self.start();self.assertEqual(second.wait(timeout=3),3)
  self.assertFalse((self.root/'protected').exists());(self.root/'gate').touch();self.assertEqual(first.wait(timeout=3),0)
  self.assertTrue((self.root/'protected').exists());self.assertEqual(self.terminal()['status'],'complete')
 def test_delayed_actual_process_observes_cancellation(self):
  p=self.start();self.wait(self.root/'ready');(self.active/'cancelled').touch();time.sleep(.1)
  self.assertIsNone(p.poll());self.assertFalse((self.active/'stopped.json').exists())
  (self.root/'gate').touch();self.assertEqual(p.wait(timeout=3),0);self.assertFalse((self.root/'protected').exists());self.assertEqual(self.terminal()['status'],'cancelled')
 def test_parent_loss_cannot_publish(self):
  parent=subprocess.Popen(['/bin/sleep','30']);self.processes.append(parent);self.owner['pid']=parent.pid;self.save()
  p=self.start();self.wait(self.root/'ready');parent.terminate();parent.wait(timeout=3);(self.root/'gate').touch()
  self.assertEqual(p.wait(timeout=3),0);self.assertFalse((self.root/'protected').exists());self.assertEqual(self.terminal()['status'],'cancelled')
 def test_expired_and_cancelled_startup_do_zero_work(self):
  for kind in ['expired','cancelled']:
   with self.subTest(kind=kind):
    self.owner['deadline']=time.time()*1000+100000
    if kind=='expired':self.owner['deadline']=time.time()*1000-1
    else:(self.active/'cancelled').touch()
    self.save();p=self.start();self.assertEqual(p.wait(timeout=3),0);self.assertFalse((self.root/'ready').exists());self.assertFalse((self.root/'protected').exists());self.assertEqual(self.terminal()['status'],'cancelled')
 def test_absent_or_mismatched_request_and_date_never_register(self):
  for request,day in [('33333333-3333-4333-8333-333333333333',self.day),(self.request,'2026-10-18')]:
   p=self.start(request,day);self.assertEqual(p.wait(timeout=3),3);self.assertFalse((self.active/'registered.json').exists())
  (self.active/'owner.json').unlink();p=self.start();self.assertEqual(p.wait(timeout=3),3);self.assertFalse((self.root/'protected').exists())
 def test_deadline_fences_running_producer(self):
  self.owner['deadline']=time.time()*1000+250;self.save();p=self.start();self.wait(self.root/'ready');time.sleep(.3);(self.root/'gate').touch()
  self.assertEqual(p.wait(timeout=3),0);self.assertFalse((self.root/'protected').exists());self.assertEqual(self.terminal()['status'],'cancelled')
if __name__=='__main__':unittest.main(verbosity=2)
