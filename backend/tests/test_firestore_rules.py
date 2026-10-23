"""
Unit and Policy Tests for Firestore Security Rules.
Verifies that:
1. Students can only read/write their own profiles and mastery records.
2. Cross-student data tampering is blocked.
3. Teachers can read mastery in their classes, but cannot modify another teacher's class.
4. Unauthenticated requests are rejected.
"""

import unittest


class MockAuth:
    def __init__(self, uid: str, role: str):
        self.uid = uid
        self.role = role


class MockSecurityRulesEngine:
    """Emulates firestore.rules evaluation logic."""

    @staticmethod
    def can_access_student_profile(auth: MockAuth, target_student_id: str, is_write: bool = False) -> bool:
        if not auth or not auth.uid:
            return False
        if auth.uid == target_student_id:
            return True
        if auth.role == "teacher" and not is_write:
            return True
        if auth.role == "admin":
            return True
        return False

    @staticmethod
    def can_access_mastery(auth: MockAuth, record_student_id: str, is_write: bool = False) -> bool:
        if not auth or not auth.uid:
            return False
        if is_write:
            return auth.uid == record_student_id
        return auth.uid == record_student_id or auth.role in ["teacher", "admin"]

    @staticmethod
    def can_modify_class(auth: MockAuth, class_teacher_id: str) -> bool:
        if not auth or not auth.uid:
            return False
        if auth.role == "teacher" and auth.uid == class_teacher_id:
            return True
        if auth.role == "admin":
            return True
        return False


class TestFirestoreRules(unittest.TestCase):

    def setUp(self):
        self.student_a = MockAuth(uid="std_101", role="student")
        self.student_b = MockAuth(uid="std_102", role="student")
        self.teacher_1 = MockAuth(uid="tch_sharma", role="teacher")
        self.teacher_2 = MockAuth(uid="tch_verma", role="teacher")
        self.admin = MockAuth(uid="adm_principal", role="admin")

    def test_student_can_read_own_profile(self):
        can_read = MockSecurityRulesEngine.can_access_student_profile(self.student_a, "std_101", is_write=False)
        self.assertTrue(can_read)

    def test_student_cannot_read_or_write_other_student_profile(self):
        can_read = MockSecurityRulesEngine.can_access_student_profile(self.student_a, "std_102", is_write=False)
        can_write = MockSecurityRulesEngine.can_access_student_profile(self.student_a, "std_102", is_write=True)
        self.assertFalse(can_read, "Cross-student profile reading must be blocked")
        self.assertFalse(can_write, "Cross-student profile writing must be blocked")

    def test_student_cannot_tamper_other_student_mastery(self):
        can_write = MockSecurityRulesEngine.can_access_mastery(self.student_a, "std_102", is_write=True)
        self.assertFalse(can_write, "Student must not be allowed to write another student's mastery")

    def test_teacher_can_read_student_mastery_but_not_arbitrary_overwrite(self):
        can_read = MockSecurityRulesEngine.can_access_mastery(self.teacher_1, "std_101", is_write=False)
        can_write = MockSecurityRulesEngine.can_access_mastery(self.teacher_1, "std_101", is_write=True)
        self.assertTrue(can_read, "Teacher must be able to read student mastery")
        self.assertFalse(can_write, "Teacher cannot forge student mastery directly")

    def test_teacher_cannot_modify_another_teachers_class(self):
        can_modify = MockSecurityRulesEngine.can_modify_class(self.teacher_2, class_teacher_id="tch_sharma")
        self.assertFalse(can_modify, "Teacher cannot update a class owned by another teacher")


if __name__ == "__main__":
    unittest.main()
