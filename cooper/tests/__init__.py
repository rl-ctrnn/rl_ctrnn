import unittest
from . import test_ctrnn

def suite():
    suite = unittest.TestSuite()
    suite.addTest(test_ctrnn.CtrnnTests('test_ctrnn'))
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
