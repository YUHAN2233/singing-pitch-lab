"""End-to-end API checks using temporary files and generated voice-like tones."""
import io
import math
import time
import unittest
import uuid
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from backend import app as service


def wav(left=220, right=330, seconds=1.1, mixed=False):
    rate=44100;t=np.arange(int(rate*seconds))/rate
    def voice(freq): return sum(.18/k*np.sin(2*np.pi*freq*k*t) for k in range(1,6))
    a,b=voice(left),voice(right)
    signal=np.stack((a+b,a+b),axis=1) if mixed else np.stack((a,b),axis=1)
    target=io.BytesIO();sf.write(target,signal,rate,format='WAV',subtype='PCM_16')
    return target.getvalue()


class DuetApiTests(unittest.TestCase):
    def setUp(self):
        scratch=Path(__file__).resolve().parents[1]/'.test-runs'
        scratch.mkdir(exist_ok=True)
        self.folder=scratch/('duet-api-'+uuid.uuid4().hex)
        self.folder.mkdir()
        self.previous=service.DATA
        service.DATA=self.folder
        with service.LOCK: service.JOBS.clear()
        self.client=TestClient(service.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None,None,None)
        with service.LOCK: service.JOBS.clear()
        service.DATA=self.previous
        shutil.rmtree(self.folder)

    def submit(self,mode,data):
        r=self.client.post('/api/jobs',params={'kind':'recording','mode':mode},content=data,
                           headers={'x-filename':'synthetic.wav','content-type':'application/octet-stream'})
        self.assertEqual(r.status_code,202,r.text)
        job=r.json();deadline=time.monotonic()+25
        while job['status'] not in ('done','error','cancelled') and time.monotonic()<deadline:
            time.sleep(.05);job=self.client.get('/api/jobs/'+job['id']).json()
        self.assertIn(job['status'],('done','error'),job)
        return job

    def test_separate_stereo_channels(self):
        job=self.submit('duet_channels',wav())
        self.assertEqual(job['status'],'done',job)
        tracks=job['result']['tracks']
        self.assertEqual(len(tracks),2)
        for track,freq in zip(tracks,(220,330)):
            observed=[f['frequency'] for f in track['frames'] if f['frequency']]
            self.assertGreater(len(observed),20)
            self.assertLess(abs(1200*math.log2(float(np.median(observed))/freq)),25)
        self.assertEqual(self.client.get('/api/audio/'+job['id']+'/original').status_code,200)
        self.assertEqual(self.client.get('/api/analysis/'+job['id']).status_code,200)

    def test_mixed_two_voices(self):
        job=self.submit('duet_mix',wav(mixed=True))
        self.assertEqual(job['status'],'done',job)
        pair=job['result']['paired_frames']
        self.assertGreater(sum(f['voice_count']==2 for f in pair)/len(pair),.75)
        self.assertIn('experimental',job['result']['algorithm'])

    def test_identical_channels_fail_then_recover(self):
        failed=self.submit('duet_channels',wav(mixed=True))
        self.assertEqual(failed['status'],'error')
        self.assertFalse((service.DATA/failed['id']).exists())
        later=self.submit('duet_channels',wav())
        self.assertEqual(later['status'],'done')

    def test_song_mode_uses_real_local_separator(self):
        if not service.MODEL.is_file(): self.skipTest('local UVR model is not installed')
        job=self.submit('duet_song',wav(mixed=True))
        self.assertEqual(job['status'],'done',job)
        self.assertEqual(len(job['result']['tracks']),2)
        self.assertEqual(job['result']['separation'],'UVR_MDXNET_Main')
        self.assertEqual(self.client.get('/api/audio/'+job['id']+'/vocals').status_code,200)
        self.assertEqual(self.client.get('/api/audio/'+job['id']+'/accompaniment').status_code,200)


if __name__=='__main__': unittest.main()
