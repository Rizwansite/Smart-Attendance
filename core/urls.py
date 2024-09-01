from django.urls import path
from .views import *



urlpatterns = [
    
   path('', homepage, name='homepage'),
   path('homepage1/', homepage1, name='homepage1'),
   path('user/<int:pk>/', user_detail, name='user_detail'),
   path('timetable/', admin_timetable, name='admin_timetable'),

   path('timetables/', timetable_list, name='timetable_list'),

   path('reset-password/ ', admin_reset_password, name='admin_reset_password'),

    path('department/<str:department_name>/semester/<int:semester_no>/attendance/',semester_attendance, name='semester_attendance'),

    
    path('register/', register_user, name='register'),

    path('specficDetail/<int:id>/', viewDetailofSpecficStudent, name='viewDetailofSpecficStudent'),
    
    path('login/', user_login, name='login'),
    path('logout/', user_logout, name='logout'),

    path('Profile/', profile, name='Profile'),
    path('deletetimetable<int:id>/', deletetimetable, name='deletetimetable'),
    # path('adminProfile/', admin_profile, name='adminProfile'),
    
    path('allUsers/', all_users, name='allUsers'),
    path('semesters/<str:department_name>/', get_semesters, name='get_semesters'),


    path('department/create/', department_create_view, name='create_department'),
    path('departments/all', department_list, name='department_list'),
    path('department/<departmentName>', specfic_department, name='specfic_department'),

    path('semester/create/', create_semester, name='create_semester'),
    path('semesters/all', display_semesters, name='display_semesters'),
    path('ttr', TRainDataSet, name='TRainDataSet'),
 
    path('course/create/', create_course, name='create_course'),
    path('course/list/', list_courses, name='list_courses'),
    # path('cameraa/', cameraa, name='cameraa'),
    # path('ca/', ca, name='ca'),
    path('train/', TrainDataSet, name='train_lbph'),
    path('recognize/', FaceRecognize, name='recognize_face'),    # Add other URL patterns for updating, deleting, etc. if needed

    path('camera', DataSet, name='camera'),
    path('teacher-registration/', create_teacher, name='teacher_registration'),
    path('allTeacher/', all_teachers, name='all_teacher'),
    # path('viewTeacherProfile/<int:cnic>', viewTeacherProfile, name='viewTeacherProfile'),
    path('teacher/profile/<str:cnic>/', viewTeacherProfile, name='viewTeacherProfile'),
      path('teacher/TA/<str:cnic>/', viewTAProfile, name='TA'),
    path('assign/', assign_course, name='assign_course'),
    path('assigned-courses/', assigned_courses, name='assigned_courses'),

    path('assigned-courses/<int:pk>/edit/', edit_assignment, name='edit_assignment'),
    path('assigned-courses/<int:pk>/delete/', delete_assignment, name='delete_assignment'),
    path('get_students/<int:department_id>/<int:course_id>/<int:semester_id>/', get_students, name='get_students'),
    path('get_students/<int:assignment_id>/', get_students_by_teacher_assignment, name='get_studentss'),
    path('upload_marks/<int:assignment_id>/', get_students_by_teacher_assignmentt, name='get_studentsss'),
    path('recognize_faces/', upload_image, name='recognize_faces'),
    path('loadface/', load_database_known_faces, name='loadface'),
    path('view_attendance/<course>/<semester>', attendance_record, name='view_attendance'),
    path('attendance/<str:course_code>/<int:semester_no>/<str:cnic>/', get_students_attendance, name='get_students_attendance'),
    path('select-attendance/', select_attendance_view, name='select_attendance'),
    path('attendance/<department_id>/<semester_id>/<course_id>/', attendance_view, name='attendance_view'),    # Other URL patterns
    path('department_lists/', department_lists, name='department_lists'),
    path('fetch_courses/', fetch_courses, name='fetch_courses'),
    path('data/', fetch_data, name='fetch_data'),
    path('marks/<int:id>/', uploadmarks, name='marks'), 
    path('viewmarks/<int:id>/', viewMarks, name='viewmarks'), 
    path('student/marks/', student_all_marks, name='student_all_marks'),
    path('attendanceSummary/<course>/<semester>/<totalLecture>', student_attendance_summary, name='student_attendance_summary'),
    path('search_date', searchByDate, name='searchByDate'),
    path('search_by_month', searchByMonth, name='searchByMonth'),
    path('count_attendance/', count_attendance, name='count_attendance'),
    path('date', searchByDateforAdmin, name='searchByDateforAdmin'),
    path('search__date', searchByDateforStudent, name='searchByDateforStudent'),
path('attendance/<int:user_id>/<str:course_code>', student_course_attendance, name='studentcourseattendance'),
 path('student-marks/<int:student_id>/', student_marks, name='student_marks'),
]