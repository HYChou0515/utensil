import unittest as ut
from importlib import reload
import os


class TestDefaultConstant(ut.TestCase):

    def setUp(self) -> None:
        self.project_root = os.path.dirname(os.path.dirname(__file__))
        try:
            self.original_project_root = os.environ['UTENSIL_PROJECT_ROOT']
        except KeyError:
            pass
        os.environ['UTENSIL_PROJECT_ROOT'] = self.project_root

    def tearDown(self) -> None:
        try:
            os.environ['UTENSIL_PROJECT_ROOT'] = self.original_project_root
        except AttributeError:
            del os.environ['UTENSIL_PROJECT_ROOT']

    def test_PROJECT_ProjectRoot(self):
        from utensil import constant
        reload(constant)
        self.assertEqual(self.project_root, constant.PROJECT['ProjectRoot'])

    def test_PROJECT_ProjectName(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('.utensil', constant.PROJECT['ProjectName'])

    def test_PROJECT_ProjectAbbr(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('.utensil', constant.PROJECT['ProjectAbbr'])

    def test_PROJECT_ProjectState(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('dev', constant.PROJECT['ProjectState'])

    def test_PROJECT_ConfigPath(self):
        from utensil import constant
        reload(constant)
        self.assertEqual(os.path.join(self.project_root, 'utensil.ini'), constant.PROJECT['ConfigPath'])

    def test_HOST_INFO_HostName(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('localhost', constant.HOST_INFO['HostName'])

    def test_LOG_Dir(self):
        from utensil import constant
        reload(constant)
        self.assertEqual(os.path.join(self.project_root, '.utensil', 'log'), constant.LOG['Dir'])

    def test_LOG_Stream(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('info', constant.LOG['Stream'])

    def test_LOG_Syslog(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('notset', constant.LOG['Syslog'])

    def test_LOG_File(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('info', constant.LOG['File'])

    def test_LOG_FilePrefix(self):
        from utensil import constant
        reload(constant)
        self.assertEqual(os.path.join(self.project_root, '.utensil', 'log', '.utensil.log'), constant.LOG['FilePrefix'])

    def test_LOG_Level(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('info', constant.LOG['Level'])

    def test_LOG_MaxMessageLen(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('60000', constant.LOG['MaxMessageLen'])


class TestModifiedConstant(ut.TestCase):
    def setUp(self) -> None:
        self.project_root = os.path.dirname(__file__)
        try:
            self.original_project_root = os.environ['UTENSIL_PROJECT_ROOT']
        except KeyError:
            pass
        try:
            self.original_utensil_config = os.environ['UTENSIL_CONFIG']
        except KeyError:
            pass
        self.utensil_config = 'utensil_test1.ini'
        os.environ['UTENSIL_CONFIG'] = self.utensil_config
        os.environ['UTENSIL_PROJECT_ROOT'] = self.project_root

    def tearDown(self) -> None:
        try:
            os.environ['UTENSIL_PROJECT_ROOT'] = self.original_project_root
        except AttributeError:
            del os.environ['UTENSIL_PROJECT_ROOT']
        try:
            os.environ['UTENSIL_CONFIG'] = self.original_utensil_config
        except AttributeError:
            del os.environ['UTENSIL_CONFIG']

    def test_PROJECT_ProjectRoot(self):
        from utensil import constant
        reload(constant)
        self.assertEqual(self.project_root, constant.PROJECT['ProjectRoot'])

    def test_PROJECT_ProjectName(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('proj_name', constant.PROJECT['ProjectName'])

    def test_PROJECT_ProjectAbbr(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('proj_name_proj_abbr', constant.PROJECT['ProjectAbbr'])

    def test_PROJECT_ProjectState(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('proj_state', constant.PROJECT['ProjectState'])

    def test_PROJECT_ConfigPath(self):
        from utensil import constant
        reload(constant)
        self.assertEqual(os.path.join(self.project_root, self.utensil_config), constant.PROJECT['ConfigPath'])

    def test_HOST_INFO_HostName(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('my_host', constant.HOST_INFO['HostName'])

    def test_LOG_Dir(self):
        from utensil import constant
        reload(constant)
        self.assertEqual(os.path.join('proj_name_proj_abbr', 'log_test'), constant.LOG['Dir'])

    def test_LOG_Stream(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('notset', constant.LOG['Stream'])

    def test_LOG_Syslog(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('info', constant.LOG['Syslog'])

    def test_LOG_File(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('notset', constant.LOG['File'])

    def test_LOG_FilePrefix(self):
        from utensil import constant
        reload(constant)
        self.assertEqual(os.path.join('proj_name_proj_abbr', 'log_test', 'proj_name.log123'), constant.LOG['FilePrefix'])

    def test_LOG_Level(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('debug', constant.LOG['Level'])

    def test_LOG_MaxMessageLen(self):
        from utensil import constant
        reload(constant)
        self.assertEqual('12345', constant.LOG['MaxMessageLen'])


if __name__ == '__main__':
    ut.main()
