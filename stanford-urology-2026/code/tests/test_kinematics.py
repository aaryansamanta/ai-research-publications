import unittest
from pelvic_motion.kinematics import calculate_kinematics,validate_track

class TestKinematics(unittest.TestCase):
    def test_constant_velocity(self):
        r=calculate_kinematics([0,1,2,3],[0,2,4,6],[0,0,0,0],0.5)
        self.assertAlmostEqual(r[-1]["dx"],3)
        self.assertTrue(all(abs(v["vx"]-1)<1e-9 for v in r))
        self.assertTrue(all(abs(v["ax"])<1e-9 for v in r))
    def test_nonuniform_quadratic(self):
        t=[0,1,3,4];r=calculate_kinematics(t,[v*v for v in t],[0]*4)
        self.assertAlmostEqual(r[1]["vx"],2)
        self.assertAlmostEqual(r[2]["vx"],6)
    def test_invalid_time(self):
        with self.assertRaises(ValueError):calculate_kinematics([0,1,1],[1,2,3],[1,2,3])
    def test_bad_scale(self):
        with self.assertRaises(ValueError):calculate_kinematics([0,1,2],[1,2,3],[1,2,3],0)

if __name__=="__main__":unittest.main()
