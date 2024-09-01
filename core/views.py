from django.shortcuts import render, redirect,HttpResponse
from django.contrib.auth import authenticate, login,logout
from .forms import UserLoginForm
from django.contrib import messages
from django.shortcuts import get_object_or_404
from .models import *
from .forms import *
import cv2 as cv
import face_recognition
import os
import numpy as np
from django.http import JsonResponse
from django.db.models import ObjectDoesNotExist
from django.conf import settings
from datetime import date
from django.urls import reverse
from django.contrib.auth.hashers import make_password
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Q
import time

from django.contrib.auth import get_user_model
User = get_user_model()

def homepage(request):
    return render(request,'index.html')
def homepage1(request):
    return redirect('login')


def register_user(request):
    if request.method == 'POST':
        form = StudentForm(request.POST, request.FILES)
        parent_form = ParentForm(request.POST, request.FILES)
        if form.is_valid() and parent_form.is_valid():
            # Save Parent user data
            parent_user_data = {
                'cnic': parent_form.cleaned_data['father__cnic'],
                'password': parent_form.cleaned_data['father__password'],
                'first_name': parent_form.cleaned_data['father__first_name'],
                'email': parent_form.cleaned_data['father__email'],
                'phone_no': parent_form.cleaned_data['father__phone_no'],
                'gender': 'M',  # Assuming all parents are male for simplicity
            }

            # Check if the parent already exists
            try:
                parent = Parent.objects.get(user__cnic=parent_user_data['cnic'])
            except Parent.DoesNotExist:
                # Create parent user and parent instance
                parent_user = User.objects.create_user(**parent_user_data)
                parent = Parent.objects.create(user=parent_user)

            # Save Student data
            user_data = {
                'cnic': form.cleaned_data['cnic'],
                'registration_no': form.cleaned_data['registration_no'],
                'password': form.cleaned_data['password'],
                'roll_no': form.cleaned_data['roll_no'],
                'first_name': form.cleaned_data['first_name'],
                'email': form.cleaned_data['email'],
                'phone_no': form.cleaned_data['phone_no'],
                'gender': form.cleaned_data['gender'],
                'image': form.cleaned_data['image'],
                'age': form.cleaned_data['age'],
                'address': form.cleaned_data['address'],
            }
            user = User.objects.create_user(**user_data)

            student_data = {
                'user': user,
                'department': form.cleaned_data['department'],
                'semester': form.cleaned_data['semester'],
                'parent': parent,
            }
            Student.objects.create(**student_data)

            messages.success(request, 'Registration successful.')
            return redirect('register')
    else:
        form = StudentForm()
        parent_form = ParentForm()
    return render(request, 'register.html', {'form': form, 'parent_form': parent_form})

def create_teacher(request):
    if request.method == 'POST':
        form = TeacherForm(request.POST)
        if form.is_valid():
            # Create a User instance
            user_data = {
                'cnic': form.cleaned_data['teacher__cnic'],
                'password': form.cleaned_data['teacher__password'],
                'first_name': form.cleaned_data['teacher__first_name'],
                'email': form.cleaned_data['teacher__email'],
                'phone_no': form.cleaned_data['teacher__phone_no'],
                'gender': 'M',  # Assuming all teachers are male for simplicity
            }
            teacher_user = User.objects.create_user(**user_data)

            # Create a Teacher instance associated with the created User
            teacher = Teacher.objects.create(user=teacher_user, department=form.cleaned_data['department'])

            # Redirect to the teacher registration page
            return redirect('teacher_registration')
    else:
        form = TeacherForm()
    return render(request, 'admin/teacher_registration.html', {'form': form})

def all_teachers(request):
    departments = Department.objects.all()
    teachers = Teacher.objects.all()

    if request.method == 'POST':
        dname = request.POST.get('departmentname')
        if dname:
            teachers = Teacher.objects.filter(department__Department_Name=dname)

    return render(request, 'admin/all_teachers.html', {'teachers': teachers, 'departments': departments})
# def viewTeacherProfile(request, cnic):
#     teacher = get_object_or_404(Teacher, user__cnic=cnic)
#     return render(request, 'admin/view_teacher_profile.html', {'teacher': teacher})

def viewTeacherProfile(request, cnic):
    teacher = get_object_or_404(Teacher, user__cnic=cnic)
    courses = Course.objects.filter(department=teacher.department)
    assignCourses = Assignment.objects.filter(teacher=teacher)
    
    for assignment in assignCourses:
        print(f"Course: {assignment.course.course_code} - {assignment.course.course_title}, Semester: {assignment.semester.Semester_No}")
    
    return render(request, 'teacher/teacherprofile.html', {
        'teacher': teacher,
        'courses': courses,
        'assigned_courses': assignCourses
    })


def viewTAProfile(request, cnic):
    teacher = get_object_or_404(Teacher, user__cnic=cnic)
    courses = Course.objects.filter(department=teacher.department)
    assignCourses = Assignment.objects.filter(teacher=teacher)
    
    for assignment in assignCourses:
        print(f"Course: {assignment.course.course_code} - {assignment.course.course_title}, Semester: {assignment.semester.Semester_No}")
    
    return render(request, 'admin/TA.html', {
        'teacher': teacher,
        'courses': courses,
        'assigned_courses': assignCourses
    })

def user_login(request):
    if request.user.is_authenticated:
        logout(request)  # End the previous session
        messages.info(request, 'You have been logged out. Please log in again.')

    login_error = False
    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            cnic = form.cleaned_data['cnic']
            password = form.cleaned_data['password']
            user = authenticate(request, cnic=cnic, password=password)
            if user is not None:
                login(request, user)
                return redirect('Profile')
            else:
                login_error = True
    else:
        form = UserLoginForm()

    return render(request, 'index.html', {'form': form, 'login_error': login_error})





def profile(request):
    if request.user.is_authenticated:
        if request.user.is_superuser:
            return render(request, 'admin/admin.html', {'user': request.user})
       
        elif hasattr(request.user, 'parent'):
            # User has a parent, so retrieve all children related to this parent
            # list_of_all_child = Student.objects.filter(parent=request.user.parent)
            children = Student.objects.filter(parent=request.user.parent)
            print('children   ',children)
            return render(request, 'student/profileStudent.html', {'user': request.user, 'children': children})
       
        elif hasattr(request.user, 'teacher'):
            teacher = request.user.teacher
            courses = Course.objects.filter(semester__DepartmentName=teacher.department)
            assignCourses = Assignment.objects.filter(teacher=teacher)
            for assignment in assignCourses:
                    print(f"Course: {assignment.course.course_code} - {assignment.course.course_title}, Semester: {assignment.semester.Semester_No}")
            
            return render(request, 'teacher/teacherprofile.html', {'user': request.user, 'courses': courses,'assigned_courses':assignCourses})
            
        else:
            # Assuming it's a student
            student = request.user.student
            courses = Course.objects.filter(semester=student.semester, department=student.department)
            return render(request, 'student/profileStudent.html', {'user': request.user, 'courses': courses})
    else:
        return redirect('login')
    
# def user_detail(request, pk):
#     user = get_object_or_404(User, pk=pk)
#     return render(request, 'admin/tta.html', {'user': user})



def user_detail(request, pk):
    user = get_object_or_404(User, pk=pk)
    student = get_object_or_404(Student, user=user)
    department_name = student.department.Department_Name
    semester_no = student.semester.Semester_No

    courses = Course.objects.filter(department__Department_Name=department_name, semester__Semester_No=semester_no)

    if request.method == 'POST':
        cnic = request.POST.get('cnic')
        registration_no = request.POST.get('registration_no')
        roll_no = request.POST.get('roll_no')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        phone_no = request.POST.get('phone_no')
        gender = request.POST.get('gender')
        address = request.POST.get('address')
        age = request.POST.get('age')
        image = request.POST.get('image')
        
        # Perform necessary validation
        if not cnic:
            messages.error(request, 'CNIC is required.')
        elif not first_name:
            messages.error(request, 'First name is required.')
        elif not last_name:
            messages.error(request, 'Last name is required.')
        elif not email:
            messages.error(request, 'Email is required.')
        elif not phone_no:
            messages.error(request, 'Phone number is required.')
        elif not gender:
            messages.error(request, 'Gender is required.')
        else:
            # Update user fields
            user.cnic = cnic
            user.registration_no = registration_no
            user.roll_no = roll_no
            user.first_name = first_name
            user.last_name = last_name
            user.email = email
            user.phone_no = phone_no
            user.gender = gender
            user.address = address
            user.age = age
            user.image = image
            
            user.save()
            messages.success(request, 'Profile updated successfully.')
            return redirect('user_detail', pk=user.pk)

    return render(request, 'admin/tta.html', {
        'user': user,
        'id': pk,
        'std_id': student,
        'courses_std': courses
    }) 


def student_course_attendance(request, user_id, course_code):
    # Ensure the Student exists
    student = get_object_or_404(Student, user_id=user_id)
    print(f"Student: {student}")

    # Ensure the Course exists
    course = get_object_or_404(Course, course_code=course_code)
    print(f"Course: {course}")

    # Fetch the attendance records for the student in the specified course
    attendance_records = Attendance.objects.filter(student=student, course=course)

    # Render the template with the attendance data
    return render(request, 'admin/attendance/student_course_attendance.html', {
        'student': student,
        'course': course,
        'attendance_records': attendance_records,
    })

def course_attendance_report(request, course_code):
    course = get_object_or_404(Course, course_code=course_code)

    attendance_records = Attendance.objects.filter(course=course)

    # Calculate total present and absent days for each student
    attendance_summary = attendance_records.values('student').annotate(
        total_present=Count('is_present', filter=Q(is_present=True)),
        total_absent=Count('is_present', filter=Q(is_present=False))
    )

    # Filter students whose attendance is below 70%
    students_below_70_percent = []
    for summary in attendance_summary:
        student = get_object_or_404(Student, pk=summary['student'])
        total_days = summary['total_present'] + summary['total_absent']
        attendance_percentage = (summary['total_present'] / total_days) * 100
        if attendance_percentage < 70:
            students_below_70_percent.append({
                'student': student,
                'total_present': summary['total_present'],
                'total_absent': summary['total_absent'],
                'attendance_percentage': attendance_percentage
            })

    return render(request, 'attendance/course_attendance_report.html', {
        'course': course,
        'students_below_70_percent': students_below_70_percent,
    })




def semester_attendance(request, department_name, semester_no):
    department = get_object_or_404(Department, Department_Name=department_name)
    semester = get_object_or_404(Semester, DepartmentName=department, Semester_No=semester_no)
    attendance_records = Attendance.objects.filter(semester=semester).select_related('student', 'student__user', 'course').order_by('student__user__first_name', 'attendance_date')

    context = {
        'department': department,
        'semester': semester,
        'attendance_records': attendance_records,
    }

    return render(request, 'admin/department/semester_attendance.html', context)

def admin_reset_password(request):
    if request.method == 'POST':
        form = AdminPasswordResetForm(request.POST)
        if form.is_valid():
            cnic = form.cleaned_data['cnic']
            new_password = form.cleaned_data['new_password']
            user = get_object_or_404(User, cnic=cnic)
            user.password = make_password(new_password)
            user.save()
            messages.success(request, 'Password updated successfully.')
            return redirect('admin_reset_password')
    else:
        form = AdminPasswordResetForm()
    
    return render(request, 'admin/reset_password.html', {'form': form})

def admin_timetable(request):
    if request.method == 'POST':
        form = TimeTableForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, 'Timetable image uploaded successfully.')
            return redirect('admin_timetable')
    else:
        form = TimeTableForm()
    
    return render(request, 'admin/timetable.html', {'form': form})

def timetable_list(request):
    timetables = TimeTable.objects.all()
    return render(request, 'admin/timetable_list.html', {'timetables': timetables})

def deletetimetable(request,id):
    timetables = TimeTable.objects.get(id=id).delete()
    return redirect('admin_timetable')
    


def get_students(request, department_id, course_id, semester_id):
 pass


def get_studentss(request, department_id):
 pass


# def all_users(request):
#     all_user = Student.objects.all()
#     return render(request,'admin/allUsers.html',{'users':all_user})


def all_users(request):
    departments = Department.objects.all()
    semesters = Semester.objects.all()

    if request.method == 'POST':
        department = request.POST.get('department')
        semester = request.POST.get('semester')

        if department and semester:
            students = Student.objects.filter(department__Department_Name=department, semester__Semester_No=semester)
        elif department:
            students = Student.objects.filter(department__Department_Name=department)
        elif semester:
            students = Student.objects.filter(semester__Semester_No=semester)
        else:
            students = Student.objects.all()
    else:
        students = Student.objects.all()

    return render(request, 'admin/allUsers.html', {'users': students, 'departments': departments, 'semesters': semesters})

def get_semesters(request, department_name):
    department = get_object_or_404(Department, Department_Name=department_name)
    semesters = department.semesters.all()
    semester_data = [{'id': semester.id, 'Semester_No': semester.Semester_No} for semester in semesters]
    return JsonResponse(semester_data, safe=False)


def user_logout(request):
    logout(request)
    return redirect('homepage')

def department_create_view(request):
    if request.method == 'POST':
        form = DepartmentForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Department created successfully.')
            return redirect('create_department')  
    else:
        form = DepartmentForm()
    return render(request, 'admin/department/add_department.html', {'form': form, 'messages': messages.get_messages(request)})

def department_list(request):
    departments = Department.objects.all()
    return render(request, 'admin/department/all_department.html', {'departments': departments})

def specfic_department(request,departmentName):
 departName = Department.objects.get(Department_Name=departmentName)
 return render(request,'admin/department/specfic_department.html',{'department':departName})
 pass

def create_semester(request):
    if request.method == 'POST':
        form = SemesterForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Semester created successfully.') 
            return redirect('create_semester')
    else:
        form = SemesterForm()
    return render(request, 'admin/semester/add_semester.html', {'form': form, 'messages': messages.get_messages(request)})

def display_semesters(request):
    semesters = Semester.objects.all().order_by('Semester_No')
    return render(request, 'admin/semester/all_semester.html', {'semesters': semesters})

def create_course(request):
    if request.method == 'POST':
        form = CourseForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Created successfully.') 
            return redirect('list_courses')
    else:
        form = CourseForm()
    return render(request, 'admin/course/add_course.html', {'form': form})

def list_courses(request):
    courses = Course.objects.all()
    return render(request, 'admin/course/all_course.html', {'courses': courses})

def face_detect_with_eye_distance(img, face_classifier, eye_classifier, focal_length=850, real_eye_distance_cm=6.3):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract face region for eye detection
        face_region_gray = gray[y:y + h, x:x + w]
        face_region_color = img[y:y + h, x:x + w]

        eyes = eye_classifier.detectMultiScale(face_region_gray)

        if len(eyes) >= 2:  # Ensure at least 2 eyes are detected
            # Sort eyes based on x-coordinate to get left and right eyes
            eyes = sorted(eyes, key=lambda e: e[0])

            # Get the coordinates of the first two eyes (left and right)
            eye1 = eyes[0]
            eye2 = eyes[1]

            # Calculate the centers of the eyes
            eye1_center = (x + eye1[0] + eye1[2] // 2, y + eye1[1] + eye1[3] // 2)
            eye2_center = (x + eye2[0] + eye2[2] // 2, y + eye2[1] + eye2[3] // 2)

            # Calculate the pixel distance between the two eyes
            eye_distance_px = np.sqrt((eye2_center[0] - eye1_center[0]) ** 2 + (eye2_center[1] - eye1_center[1]) ** 2)

            # Convert the pixel distance to centimeters using the focal length and real distance
            eye_distance_cm = (real_eye_distance_cm * focal_length) / eye_distance_px

            # Draw circles around the eyes and a line connecting them
            cv.circle(img, eye1_center, 5, (0, 255, 0), 2)
            cv.circle(img, eye2_center, 5, (0, 255, 0), 2)
            cv.line(img, eye1_center, eye2_center, (0, 255, 0), 2)

            # Display eye distance in cm
            cv.putText(img, f"Eye Distance: {eye_distance_cm:.2f} cm", (x, y - 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img

def face_extract(img, face_classifier):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    extracted_faces = []
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face_resized = cv.resize(face, (300, 300))  # Ensure consistent size
        extracted_faces.append(face_resized)
    return extracted_faces

def face_detect(img, face_classifier):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    detected_faces = []
    face_coordinates = []
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        face_resized = cv.resize(face, (200, 200))
        detected_faces.append(face_resized)
        face_coordinates.append((x, y))
    return img, detected_faces, face_coordinates


  # Import time for delays

def DataSet(request):
    if request.method == 'POST':
        cascadeClassifierPath = os.path.join(os.getcwd(),'core/templates/cascadeClassifier')
        cascade_path = os.path.join(cascadeClassifierPath, 'hasscode_classifire_frontalFace.xml')
        face_classifier = cv.CascadeClassifier(cascade_path)

        if face_classifier.empty():
            return HttpResponse('Error loading classifier')

        cap = cv.VideoCapture(0)
        count = 0
        student_id = request.POST.get('student_id')
        if not student_id:
            return HttpResponse("Please provide a valid student ID.")

        while count < 200:
            ret, frame = cap.read()
            if not ret:
                continue

            image, faces, coordinates = face_detect(frame, face_classifier)
            if len(faces) > 0:
                for face in faces:
                    count += 1
                    face_resized = cv.resize(face, (300, 300))
                    file_name_path = os.path.join(os.getcwd(), 'core/templates/dataset')
                    os.makedirs(file_name_path, exist_ok=True)
                    file_name = f"student_{student_id}_{count}.jpg"
                    file_path = os.path.join(file_name_path, file_name)
                    cv.imwrite(file_path, face_resized, [cv.IMWRITE_JPEG_QUALITY, 100])
                    cv.putText(face_resized, str(count), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv.imshow('Face Cropper', face_resized)
                    time.sleep(0.1)  # Add a small delay between captures
            else:
                print('Face Not Found')

            if cv.waitKey(1) == 13:  # Break on Enter key
                break

        cap.release()
        cv.destroyAllWindows()
        return HttpResponse('Data collection completed successfully.')
    else:
        return render(request, 'camera.html')

def TrainDataSet(request):
    data_path = os.path.join(os.getcwd(), 'core/templates/dataset')
    onlyfiles = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    Training_Data = []
    Labels = []

    for file in onlyfiles:
        student_id = file.split('_')[1]
        image_path = os.path.join(data_path, file)
        face_img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if face_img is not None:
            face_resized = cv.resize(face_img, (300, 300))  # Consistent resizing
            Training_Data.append(np.asarray(face_resized, dtype=np.uint8))
            Labels.append(int(student_id))

    Labels = np.asarray(Labels, dtype=np.int64)
    model = cv.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    model_path = os.path.join(os.getcwd(), 'core/templates/trainedModel/trained_model.yml')
    model.save(model_path)

    return HttpResponse('Model Trained')

def FaceRecognize(request):
    try:
        model_path = os.path.join(os.getcwd(), 'core/templates/trainedModel/trained_model.yml')
        model = cv.face.LBPHFaceRecognizer_create()
        model.read(model_path)

        cascadeClassifierPath = os.path.join(os.getcwd(), 'core/templates/cascadeClassifier')
        cascade_path = os.path.join(cascadeClassifierPath, 'hasscode_classifire_frontalFace.xml')

        if not os.path.exists(cascade_path):
            return JsonResponse({'error': 'Cascade file does not exist'}, status=500)

        face_classifier = cv.CascadeClassifier(cascade_path)
        eye_classifier = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

        if face_classifier.empty() or eye_classifier.empty():
            return JsonResponse({'error': 'Error loading classifiers'}, status=500)

        cap = cv.VideoCapture(0)
        recognized_ids = []

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            cv.imshow('Press T to capture', frame)

            if cv.waitKey(1) & 0xFF == ord('t'):
                # Detect face and eye distance
                frame_with_eye_distance = face_detect_with_eye_distance(frame, face_classifier, eye_classifier)
                image, faces, coordinates = face_detect(frame, face_classifier)

                if len(faces) > 0:
                    for i, face in enumerate(faces):
                        face_gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
                        res = model.predict(face_gray)
                        x, y = coordinates[i]
                        if res[1] < 500:  # Adjust threshold based on testing
                            student_id = res[0]
                            confidence = int(100 * (1 - (res[1]) / 300))

                            if confidence > 82:  # Adjust this threshold based on testing
                                recognized_ids.append(student_id)
                                cv.putText(image, f"ID: {student_id}", (x, y - 30), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                                
                                # Fetch semester information
                                try:
                                    student = Student.objects.get(user__roll_no=student_id)
                                    semester = student.semester.Semester_No
                                    cv.putText(image, f"Sem: {semester}", (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                                except Student.DoesNotExist:
                                    cv.putText(image, 'Unknown Student', (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                            else:
                                cv.putText(image, 'Unknown', (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                        else:
                            cv.putText(image, 'Unknown', (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

                    cv.imshow('Face Recognizer', frame_with_eye_distance)
                else:
                    cv.putText(frame_with_eye_distance, 'Face not found', (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    cv.imshow('Face Recognizer', frame_with_eye_distance)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

        unique_recognized_ids = set(recognized_ids)
        students_data = Student.objects.filter(user__roll_no__in=unique_recognized_ids)

        id = request.session.get('assi')
        if not id:
            return JsonResponse({'error': 'No assignment found in session'}, status=400)
        
        assignment = get_object_or_404(Assignment, pk=id)
        
        all_students = Student.objects.filter(department=assignment.course.department, semester=assignment.semester)
        
        for student in all_students:
            is_present = student in students_data
            Attendance.objects.update_or_create(
                student=student,
                attendance_date=date.today(),
                course=assignment.course,
                semester=assignment.semester,
                defaults={'is_present': is_present}
            )

        return render(request, 'teacher/matchedSTD.html', {'students_data': students_data})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# def DataSet(request):
#  if request.method == 'POST':
      
#       cascadeClassifierPath = os.path.join(os.getcwd(),'core/templates/cascadeClassifier')
#       cascade_path = os.path.join(cascadeClassifierPath, 'hasscode_classifire_frontalFace.xml')
#       face_classifier = cv.CascadeClassifier(cascade_path)
     
     
#       if face_classifier.empty():
#             print('Error loading classifier')
#             return HttpResponse('Error loading classifier')
      
#       print('Classifier loaded successfully')


#       def face_extract(img):
#         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#         faces = face_classifier.detectMultiScale(gray, 1.3, 5)

#         if faces is():
#          return None
    
#         for (x, y, w, h) in faces:
#          cropped_img = img[y:y+h, x:x+w]

#          return cropped_img
      

#       cap = cv.VideoCapture(0)
#       count = 0


#       student_id = request.POST.get('student_id')
#       if not student_id:
#             return HttpResponse("Please provide a valid student ID.")

    
#       while True:
#             ret, frame = cap.read()
#             if ret:
#                 if face_extract(frame) is not None:
#                     count = count + 1
#                     face = cv.resize(face_extract(frame), (300, 300))
#                     face = cv.cvtColor(face, cv.COLOR_BGR2GRAY) 

              

#                     file_name_path = os.path.join(os.getcwd(), 'core/templates/dataset')
#                     os.makedirs(file_name_path, exist_ok=True)

#                     file_name = f"student_{student_id}_{count}.jpg"
#                     file_path = os.path.join(file_name_path, file_name)


#                     cv.imwrite(file_path, face)
#                     cv.putText(face, str(count), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#                     cv.imshow('Face Cropper', face)
#                 else:
#                     print('Face Not Found')

#                 if cv.waitKey(1) == 13 or count == 200:
#                     break
#             else:
#                 print("Failed to capture frame")
       
#       cap.release()
#       cv.destroyAllWindows()
#       print('Data Collection Completed')
#       return HttpResponse('Data collection completed successfully.')
    
#  else:
#         return render(request,'camera.html')


# def TrainDataSet(request):
#     data_path = os.path.join(os.getcwd(), 'core/templates/dataset')
#     onlyfiles = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

#     Training_Data = []
#     Labels = []

#     # Loop through all images
#     for file in onlyfiles:
#         # Extract the student ID from the filename
#         student_id = file.split('_')[1]

#         # Read the image and convert it to grayscale
#         image_path = os.path.join(data_path, file)
#         images = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

#         # Convert the image to numpy array and append it to the training data
#         Training_Data.append(np.asarray(images, dtype=np.uint8))

#         # Append the student ID to the labels list
#         Labels.append(int(student_id))

#     # Convert labels to numpy array
#     Labels = np.asarray(Labels, dtype=np.int64)

#     # Create LBPH recognizer
#     model = cv.face.LBPHFaceRecognizer_create()

#     # Train the model with the training data and labels
#     model.train(np.asarray(Training_Data), np.asarray(Labels))

#     # Save the trained model in the 'models' directory
#     model_path = os.path.join(os.getcwd(), 'core/templates/trainedModel/trained_model.yml')
#     model.save(model_path)

#     print('Model Trained Successfully')

#     return HttpResponse('Model Trained')

def TRainDataSet(request):
    return render(request, 'admin/trainModel.html')


# def FaceRecgnize(request):
#     # Load training data and train the model
#     data_path = os.path.join(os.getcwd(), 'core/templates/dataset')
#     onlyfiles = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

#     Training_Data = []
#     Labels = []

#     for i, file in enumerate(onlyfiles):
#         image_path = os.path.join(data_path, onlyfiles[i])
#         images = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
#         Training_Data.append(np.asarray(images, dtype=np.uint8))
#         # Extract student ID from the filename
#         student_id = int(file.split('_')[1])  # Extracting the ID from the filename
#         Labels.append(student_id)

#     Labels = np.asarray(Labels, dtype=np.int32)
#     model = cv.face.LBPHFaceRecognizer_create()
#     model.train(np.asarray(Training_Data), np.asarray(Labels))

#     print('Model Trained')

#     # Load face classifier
#     cascadeClassifierPath = os.path.join(os.getcwd(), 'core/templates/cascadeClassifier')
#     cascade_path = os.path.join(cascadeClassifierPath, 'hasscode_classifire_frontalFace.xml')
#     face_classifier = cv.CascadeClassifier(cascade_path)
     
#     if face_classifier.empty():
#         print('Error loading classifier')
#         exit()

#     print('Classifier loaded successfully')

#     # Function to detect faces
#     def face_detect(img, size=0.5):
#          gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#          faces = face_classifier.detectMultiScale(gray, 1.3, 5)

#          if faces is ():
#               return img, []

#          detected_faces = []

#          for (x, y, w, h) in faces:
#                cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#                roi = img[y:y+h,x:x+w]
#                roi = cv.resize(roi,(200,200))
#                detected_faces.append(roi)

#          return img, detected_faces
    
#     cap = cv.VideoCapture(0)
#     recognized_ids = []

#     while True:
#         ret, frame = cap.read()
#         image, faces = face_detect(frame)

#         try:
#             for face in faces:
#                 face_gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
#                 res = model.predict(face_gray)

#                 if res[1] < 500:
#                     student_id = res[0]  # Recognized student ID
#                     confidence = int(100 * (1 - (res[1]) / 300))

#                     if confidence > 82:
#                         recognized_ids.append(student_id)

#                         # Display the recognized student ID
#                         cv.putText(image, str(student_id), (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#                     else:
#                         cv.putText(image, 'Unknown', (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

#             cv.imshow('Face Cropper', image)

#         except:
#             cv.putText(image, 'Face not found', (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#             cv.imshow('Face Cropper', image)
#             pass

#         if cv.waitKey(1) == 13:
#             break

#     cap.release()
#     cv.destroyAllWindows()

#     # Return the recognized IDs as JSON response
#     unique_recognized_ids = set(recognized_ids)
#     return JsonResponse({'recognized_ids': list(unique_recognized_ids)})



# def FaceRecognize(request):
#     # Load training data and train the model
#     data_path = os.path.join(os.getcwd(), 'core/templates/dataset')
#     onlyfiles = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

#     Training_Data = []
#     Labels = []

#     for i, file in enumerate(onlyfiles):
#         image_path = os.path.join(data_path, onlyfiles[i])
#         images = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
#         Training_Data.append(np.asarray(images, dtype=np.uint8))
#         student_id = int(file.split('_')[1])  # Extracting the ID from the filename
#         Labels.append(student_id)

#     Labels = np.asarray(Labels, dtype=np.int32)
#     model = cv.face.LBPHFaceRecognizer_create()
#     model.train(np.asarray(Training_Data), np.asarray(Labels))

#     print('Model Trained')

#     # Load face classifier
#     cascadeClassifierPath = os.path.join(os.getcwd(), 'core/templates/cascadeClassifier')
#     cascade_path = os.path.join(cascadeClassifierPath, 'hasscode_classifire_frontalFace.xml')
#     face_classifier = cv.CascadeClassifier(cascade_path)

#     if face_classifier.empty():
#         print('Error loading classifier')
#         exit()

#     print('Classifier loaded successfully')

#     # Function to detect faces
#     def face_detect(img, size=0.5):
#         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#         faces = face_classifier.detectMultiScale(gray, 1.3, 5)

#         if len(faces) == 0:
#             return img, []

#         detected_faces = []

#         for (x, y, w, h) in faces:
#             cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             roi = img[y:y + h, x:x + w]
#             roi = cv.resize(roi, (200, 200))
#             detected_faces.append(roi)

#         return img, detected_faces

#     cap = cv.VideoCapture(0)
#     recognized_ids = []

#     while True:
#         ret, frame = cap.read()
#         cv.imshow('Press T to capture', frame)

#         if cv.waitKey(1) & 0xFF == ord('t'):
#             image, faces = face_detect(frame)

#             if len(faces) > 0:
#                 for face in faces:
#                     face_gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
#                     res = model.predict(face_gray)

#                     if res[1] < 500:
#                         student_id = res[0]  # Recognized student ID
#                         confidence = int(100 * (1 - (res[1]) / 300))

#                         if confidence > 82:
#                             recognized_ids.append(student_id)

#                             # Display the recognized student ID
#                             cv.putText(image, str(student_id), (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#                         else:
#                             cv.putText(image, 'Unknown', (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

#                     cv.imshow('Face Recognizer', image)

#             else:
#                 cv.putText(image, 'Face not found', (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#                 cv.imshow('Face Recognizer', image)

#         if cv.waitKey(1) & 0xFF == 13:  # Press Enter to exit
#             break

#     cap.release()
#     cv.destroyAllWindows()

#     # Return the recognized IDs as JSON response
#     unique_recognized_ids = set(recognized_ids)
#     return JsonResponse({'recognized_ids': list(unique_recognized_ids)})

student_data = []
studenttts=''

# from django.views.decorators.csrf import csrf_exempt

# @csrf_exempt
# def FaceRecognize(request): 
#     try:
#         # Load training data and train the model
#         data_path = os.path.join(os.getcwd(), 'core/templates/dataset')
#         onlyfiles = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

#         Training_Data = []
#         Labels = []

#         for i, file in enumerate(onlyfiles):
#             image_path = os.path.join(data_path, onlyfiles[i])
#             images = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
#             Training_Data.append(np.asarray(images, dtype=np.uint8))
#             student_id = int(file.split('_')[1])  # Extracting the ID from the filename
#             Labels.append(student_id)

#         Labels = np.asarray(Labels, dtype=np.int64)
#         model = cv.face.LBPHFaceRecognizer_create()
#         model.train(np.asarray(Training_Data), np.asarray(Labels))

#         print('Model Trained')

#         # Load face classifier
#         cascadeClassifierPath = os.path.join(os.getcwd(), 'core/templates/cascadeClassifier')
#         cascade_path = os.path.join(cascadeClassifierPath, 'hasscode_classifire_frontalFace.xml')
        
#         # Log the path to ensure it's correct
#         print(f"Loading classifier from path: {cascade_path}")

#         # Check if the cascade file exists
#         if not os.path.exists(cascade_path):
#             print("Cascade file does not exist at path:", cascade_path)
#             return JsonResponse({'error': 'Cascade file does not exist'}, status=500)

#         face_classifier = cv.CascadeClassifier(cascade_path)

#         if face_classifier.empty():
#             print('Error loading classifier')
#             return JsonResponse({'error': 'Error loading classifier'}, status=500)

#         print('Classifier loaded successfully')

#         # Function to detect faces
#         def face_detect(img):
#             gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#             faces = face_classifier.detectMultiScale(gray, 1.3, 5)

#             if len(faces) == 0:
#                 return img, [], []

#             detected_faces = []
#             face_coordinates = []

#             for (x, y, w, h) in faces:
#                 cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 roi = img[y:y + h, x:x + w]
#                 roi = cv.resize(roi, (200, 200))
#                 detected_faces.append(roi)
#                 face_coordinates.append((x, y))

#             return img, detected_faces, face_coordinates

#         cap = cv.VideoCapture(0)
#         recognized_ids = []

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 continue

#             cv.imshow('Press T to capture', frame)

#             if cv.waitKey(1) & 0xFF == ord('t'):
#                 image, faces, coordinates = face_detect(frame)

#                 if len(faces) > 0:
#                     for i, face in enumerate(faces):
#                         face_gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
#                         res = model.predict(face_gray)
#                         x, y = coordinates[i]

#                         if res[1] < 500:
#                             student_id = res[0]  # Recognized student ID
#                             confidence = int(100 * (1 - (res[1]) / 300))

#                             if confidence > 82:
#                                 recognized_ids.append(student_id)
#                                 # Display the recognized student ID
#                                 cv.putText(image, str(student_id), (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
#                             else:
#                                 cv.putText(image, 'Unknown', (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

#                     cv.imshow('Face Recognizer', image)

#                 else:
#                     cv.putText(image, 'Face not found', (250, 450), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#                     cv.imshow('Face Recognizer', image)

#             if cv.waitKey(1) & 0xFF == ord('q'):  # Press Q to exit
#                 break

#         cap.release()
#         cv.destroyAllWindows()

#         # Return the recognized IDs as JSON response
#         unique_recognized_ids = set(recognized_ids)
#         print('IDs ',unique_recognized_ids)

#         students_data = Student.objects.filter(user__roll_no__in=unique_recognized_ids)
#         print('student data',students_data)

#         id = request.session.get('assi')
#         print(
#             'Assigned Id',id
#         )
#         if not id:
#             return JsonResponse({'error': 'No assignment found in session'}, status=400)
        
#         assignment = get_object_or_404(Assignment, pk=id)
#         print('Assignment ',assignment)
        
#         all_students = Student.objects.filter(department=assignment.course.department, semester=assignment.semester)
#         print('All students  ',all_students)
        
#         for student in all_students:
#             is_present = student in students_data
#             Attendance.objects.update_or_create(
#                 student=student,
#                 attendance_date=date.today(),
#                 course=assignment.course,
#                 semester=assignment.semester,
#                 defaults={'is_present': is_present}
#             )

#         return render(request, 'teacher/matchedSTD.html', {'students_data': students_data})



#         return JsonResponse({'recognized_ids': list(unique_recognized_ids)})

#     except Exception as e:
#         print(f"Error: {e}")
#         return JsonResponse({'error': str(e)}, status=500)
    







def assign_course(request):
    if request.method == 'POST':
        form = AssignmentForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('assign_course')
    else:
        form = AssignmentForm()
    return render(request, 'admin/assign_course.html', {'form': form})

def assigned_courses(request):
    assigned_courses = Assignment.objects.all()
    return render(request, 'admin/assigned_courses.html', {'assigned_courses': assigned_courses})

def edit_assignment(request, pk):
    try:
        assignment = Assignment.objects.get(pk=pk)
    except Assignment.DoesNotExist:
        return redirect('assigned_courses')

    if request.method == 'POST':
        form = AssignmentForm(request.POST, instance=assignment)
        if form.is_valid():
            form.save()
            return redirect('assigned_courses')
    else:
        form = AssignmentForm(instance=assignment)
    return render(request, 'admin/edit_assignment.html', {'form': form})

def delete_assignment(request, pk):
    try:
        assignment = Assignment.objects.get(pk=pk)
    except Assignment.DoesNotExist:
        return redirect('assigned_courses')

    if request.method == 'POST':
        assignment.delete()
        return redirect('assigned_courses')

    return render(request, 'delete_assignment.html', {'assignment': assignment})

def load_known_faces(student):
    try:
        student_image_path = student.user.image.path  # Get the file path of the image
        print('patt  ',student_image_path)
        
        # Load the student's image file
        student_image = face_recognition.load_image_file(student_image_path)
        print(student_image)
        
        # Encode all faces found in the student's image
        student_face_encodings = face_recognition.face_encodings(student_image)
        
        # Prepare encodings for usage
        encoded_faces = [encoding.tolist() for encoding in student_face_encodings]
        
        return encoded_faces
    except ObjectDoesNotExist:
        return []  # Handle case where image is not found
    except Exception as e:
        # Log the exception (optional, requires proper logging setup)
        print(f"Error encoding faces for student {student.user.roll_no}: {e}")
        return []





def get_students_by_teacher_assignmentt(request, assignment_id):
    assignment = get_object_or_404(Assignment, pk=assignment_id)
    student_list = Student.objects.filter(semester=assignment.semester, department=assignment.course.department)
    
    if request.method == 'POST':
        formset = MarksFormSet(request.POST)
        if formset.is_valid():
            for form in formset:
                marks = form.save(commit=False)
                marks.teacher = assignment.teacher
                marks.course = assignment.course
                marks.semester = assignment.semester
                marks.save()
            return redirect('success_url')  # Replace with your success URL
    else:
        initial_data = [{'student': student.id} for student in student_list]
        formset = MarksFormSet(queryset=Marks.objects.none(), initial=initial_data)
    
    student_data = []
    for student in student_list:
        student_data.append({
            'student_roll_no': student.user.roll_no,
            'student_name': f"{student.user.first_name} {student.user.last_name}",
            'student_image': student.user.image.url if student.user.image else None,
        })

    return render(request, 'teacher/uploadmarks.html', {
        'students': student_data,
        'formset': formset,
        'assignment': assignment,
    })

def get_students_by_teacher_assignment(request, assignment_id):
    assignment = get_object_or_404(Assignment, pk=assignment_id)
    print('Assignment ID:', assignment)
    request.session['assi'] = assignment.id

    # Get all students associated with the assignment's semester and department
    student_list = Student.objects.filter(semester=assignment.semester, department=assignment.course.department)
    print('Student data:', student_list)

    # Prepare data to return
    student_data = []
    for student in student_list:
        student_data.append({
            'student_roll_no': student.user.roll_no,
            'student_name': f"{student.user.first_name} {student.user.last_name}",
            'student_image': student.user.image.url if student.user.image else None,
        })

    return render(request, 'teacher/specficStudent.html', {'students': student_data})


def load_database_known_faces(request):
    assignment_id = request.session.get('assi')
    if not assignment_id:
        return JsonResponse({'error': 'No assignment found in session'}, status=400)
    
    assignment = get_object_or_404(Assignment, pk=assignment_id)
    student_list = Student.objects.filter(semester=assignment.semester, department=assignment.course.department)
    
    person_face_encodings = []
    for profile in student_list:
        person_image_path = os.path.join(settings.MEDIA_ROOT, str(profile.user.image))
        try:
            image_of_person = face_recognition.load_image_file(person_image_path)
            encodings = face_recognition.face_encodings(image_of_person)
            if encodings:
                person_face_encodings.append((profile.user.roll_no, encodings[0]))  # Assuming one face per image
        except FileNotFoundError:
            print(f"Error processing image for {profile.user.roll_no}: File not found")
        except Exception as e:
            print(f"Error processing image for {profile.user.roll_no}: {e}")

    return person_face_encodings
 

def uploadmarks(request, id):
    assignment = get_object_or_404(Assignment, id=id)
    print('Assignment:', assignment)
    students = Student.objects.filter(department=assignment.course.department, semester=assignment.semester)
    print('Students:', students)

    if request.method == 'POST':
        total_marks = request.POST.get('total_marks')
        print('Total Marks:', total_marks)
        if total_marks:
            for student in students:
                marks_obtained = request.POST.get(f'marks_obtained_{student.id}')
                print(f'Marks obtained for student {student.id}:', marks_obtained)
                if marks_obtained is not None:
                    Marks.objects.update_or_create(
                        student=student,
                        course=assignment.course,
                        semester=assignment.semester,
                        defaults={
                            'teacher': assignment.teacher,
                            'marks_obtained': marks_obtained,
                            'total_marks': total_marks,
                        }
                    )
            return render(request, 'teacher/uploadmarks.html', {'assignment': assignment, 'students': students, 'message': 'Marks updated successfully'})

    context = {
        'assignment': assignment,
        'students': students,
    }
    print('Data   ',context['students'])
    
    return render(request, 'teacher/uploadmarks.html', context)

def viewMarks(request, id):
    assignment = get_object_or_404(Assignment, id=id)
    
    # Extracting related objects
    teacher_name = assignment.teacher.user.first_name
    course_code = assignment.course.course_code
    course_title = assignment.course.course_title
    semester_name = assignment.semester.Semester_No

    # Filter marks by course and semester
    marks = Marks.objects.filter(course=assignment.course, semester=assignment.semester)

    # Prepare data for each student
    students_marks = [
        {
            'student_cnic': mark.student.user.cnic,
            'student_name': mark.student.user.first_name,
            'marks_obtained': mark.marks_obtained,
            'total_marks': mark.total_marks,
        }
        for mark in marks
    ]
    
    # Prepare context to pass to template
    context = {
        'teacher': teacher_name,
        'course_code': course_code,
        'course_title': course_title,
        'semester': semester_name,
        'students_marks': students_marks,
    }
    
    return render(request, 'teacher/viewMarks.html', context)

def marks(req,id):
 print(req)
 pass


# Function to extract faces from an image
def face_extract(img, face_classifier):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    extracted_faces = []
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face_resized = cv.resize(face, (300, 300))
        extracted_faces.append(face_resized)
    return extracted_faces

# Function to detect faces and get coordinates
def face_detect(img, face_classifier):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    detected_faces = []
    face_coordinates = []
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face_resized = cv.resize(face, (200, 200))
        detected_faces.append(face_resized)
        face_coordinates.append((x, y))
    return img, detected_faces, face_coordinates

# Function to match faces with known encodings
def match_faces(uploaded_face_encodings, known_face_encodings):
    matched_ids = []
    for face_encoding in uploaded_face_encodings:
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(distances) == 0:
            continue
        best_match_index = np.argmin(distances)
        if distances[best_match_index] < 0.6:  # 0.6 is the default threshold; adjust based on testing
            matched_ids.append(best_match_index)
    return matched_ids

def upload_image(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        # savedimage.create(upload_image)

        if not uploaded_file:
            return JsonResponse({'message': 'Please upload a file'}, status=400)

        # Load known face encodings from the database
        known_face_encodings = load_database_known_faces(request)
        if not isinstance(known_face_encodings, list):
            return known_face_encodings  # Error response from load_database_known_faces

        # Load face classifier
        cascadeClassifierPath = os.path.join(os.getcwd(),'core/templates/cascadeClassifier')
        cascade_path = os.path.join(cascadeClassifierPath, 'hasscode_classifire_frontalFace.xml')
        face_classifier = cv.CascadeClassifier(cascade_path)
        if face_classifier.empty():
            return JsonResponse({'error': 'Error loading face classifier'}, status=500)

        # Read and decode uploaded image
        image_content = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        img = cv.imdecode(image_content, cv.IMREAD_COLOR)

        # Detect and extract faces from the image
        _, detected_faces, face_coordinates = face_detect(img, face_classifier)
        if not detected_faces:
            return JsonResponse({'message': 'No faces detected'}, status=400)

        # Encode the detected faces
        uploaded_face_encodings = [face_recognition.face_encodings(face)[0] for face in detected_faces if face_recognition.face_encodings(face)]
        if not uploaded_face_encodings:
            return JsonResponse({'message': 'No face encodings found'}, status=400)

        # Match faces and get student IDs
        matched_ids = match_faces(uploaded_face_encodings, known_face_encodings)
        students_data = Student.objects.filter(user__roll_no__in=matched_ids)

        # Check assignment from session
        assignment_id = request.session.get('assi')
        if not assignment_id:
            return JsonResponse({'error': 'No assignment found in session'}, status=400)

        assignment = get_object_or_404(Assignment, pk=assignment_id)
        all_students = Student.objects.filter(department=assignment.course.department, semester=assignment.semester)

        # Update attendance count for the course
        course_attendance_count, created = CourseAttendanceCount.objects.get_or_create(
            course=assignment.course,
            semester=assignment.semester
        )
        course_attendance_count.count += 1
        course_attendance_count.save()

        print('Counted ', course_attendance_count.count)

        # Mark attendance for all students
        for student in all_students:
            is_present = student in students_data
            Attendance.objects.update_or_create(
                student=student,
                attendance_date=date.today(),
                course=assignment.course,
                semester=assignment.semester,
                defaults={'is_present': is_present}
            )

        return render(request, 'teacher/matchedSTD.html', {'students_data': students_data, 'attendance_count': course_attendance_count.count})

    return render(request, 'teacher/studentAttendance.html')




@login_required
def student_all_marks(request):
    user = request.user
    student = get_object_or_404(Student, user=user)
    
    # Fetch all marks for the student
    marks = Marks.objects.filter(student=student).select_related('course', 'semester')

    context = {
        'student': student,
        'marks': marks,
    }

    # return JsonResponse(context)
    
    return render(request, 'student/viewmarks.html', context)


def student_marks(request, student_id):
    print(student_id)
    student = Student.objects.get(user__pk=student_id)
    print(' sdfsf',student)
    marks = Marks.objects.filter(student=student)

    print(marks)
    return render(request, 'student/student_marks.html', {'student': student, 'marks': marks})


def match_faces(uploaded_face_encodings, known_face_encodings):
    matched_ids = []

    for uploaded_face_encoding in uploaded_face_encodings:
        found_match = False

        for roll_no, known_face_encoding in known_face_encodings:
            matches = face_recognition.compare_faces([known_face_encoding], uploaded_face_encoding, tolerance=0.6)
            if any(matches):
                matched_ids.append(roll_no)
                found_match = True
                print(f'Matched ------> {roll_no}')
                break

        if not found_match:
            print('No match found for this face')

    return matched_ids

def get_attendance(request, course_code, semester_no):
    pass

def get_students_attendance(request, course_code, semester_no, cnic):
    # Get the student object, ensuring that exactly one object matches the criteria
    student = get_object_or_404(Student, user__cnic=cnic)

    # Filter the attendance records for the retrieved student
    attendances = Attendance.objects.filter(student=student, course__course_code=course_code, semester__Semester_No=semester_no)

    # Print the attendance records for debugging purposes
    print('Student attendances:', attendances)
    return render(request, 'student/attendance_history.html',
                   {'attendance_records':attendances,
                                                               
            'course_code':course_code,
            'semester_no':semester_no                                                   })

    # Pass the attendance data to the context or return as needed
    # ...

    # For demonstration purposes, we'll just pass here
    
def student_attendance_detail(request, student_id, course_code, semester_id):
    student = get_object_or_404(Student, id=student_id)
    course = get_object_or_404(Course, course_code=course_code)
    semester = get_object_or_404(Semester, id=semester_id).first()



    attendances = Attendance.objects.filter(student=student, course=course, semester=semester)

    context = {
        'student': student,
        'course': course,
        'semester': semester,
        'attendances': attendances,
    }

    print(context)
    return JsonResponse({'data':context})

def attendance_record(request,course,semester):
#  assignment = get_object_or_404(Assignment, pk=id)

 attendance_records = Attendance.objects.filter(course__course_code=course,semester__Semester_No=semester)
 getCount = CourseAttendanceCount.objects.get(course__course_code=course,semester__Semester_No=semester)
 print('Counted ',getCount)


#  print('record  ',attendance_records)
 return render(request,'teacher/attendance_record.html',{'records':attendance_records,'course':course,'semester':semester,'getCount':getCount})

def student_attendance_summary(request, course, semester, totalLecture):
    # Get attendance records for the specific course and semester
    attendance_records = Attendance.objects.filter(course__course_code=course, semester__Semester_No=semester)
    
    # Get attendance counts for each student
    attendance_counts = attendance_records.values(
        'student__user__cnic',
        'student__user__first_name',
        'student__user__last_name'
    ).annotate(
        present_count=Count('id', filter=Q(is_present=True)),
        absent_count=Count('id', filter=Q(is_present=False)),
    )
    
    # Calculate the attendance percentage for each student
    for record in attendance_counts:
       totalLecture = int(totalLecture)  # Convert totalLecture to an integer
       record['percentage'] = (record['present_count'] / totalLecture) * 100 if totalLecture > 0 else 0

    context = {
        'course': course,
        'semester': semester,
        'total_lecture': totalLecture,
        'attendance_counts': attendance_counts,
    }
    
    return render(request, 'teacher/attendance_summary.html', context)

def select_attendance_view(request):
    if request.method == 'POST':
        form = AttendanceSelectionForm(request.POST)
        if form.is_valid():
            department = form.cleaned_data['department']
            semester = form.cleaned_data['semester']
            course = form.cleaned_data['course']

            print(department, semester,course)
            return redirect(reverse('attendance_view', args=[department, semester, course]))
    else:
        form = AttendanceSelectionForm()

    return render(request, 'admin/attendance/select_attendance.html', {'form': form})

def attendance_view(request, department_id, semester_id, course_id):
    department = get_object_or_404(Department, pk=department_id)
    print('depart  ',department)
    semester = get_object_or_404(Semester, pk=semester_id)
    print('semester  ',semester)
    course = get_object_or_404(Course, pk=course_id)
    print('course  ',course)
    students = Student.objects.filter(department=department, semester=semester)
    attendance_records = Attendance.objects.filter(course=course, semester=semester).select_related('student')
    
    attendance_data = {}
    for record in attendance_records:
        student_id = record.student.id
        if student_id not in attendance_data:
            attendance_data[student_id] = {
                'student': record.student,
                'attendance': []
            }
        attendance_data[student_id]['attendance'].append(record)

    context = {
        'department': department,
        'semester': semester,
        'course': course,
        'students': students,
        'attendance_data': attendance_data
    }
    
    return render(request, 'admin/attendance/attendance_sheet.html', context)

def department_lists(request):
    departments = Department.objects.all()
    print('departments   ',departments)
    return render(request, 'admin/attendance/select_attendance.html', {'departments': departments})
    
def fetch_courses(request):
    department_name = request.GET.get('department_name')
    semester_no = request.GET.get('semester_no')
    
    if department_name and semester_no:
        courses = Course.objects.filter(department__Department_Name=department_name, semester__Semester_No=semester_no)
        courses_data = [{'course_code': course.course_code, 'course_title': course.course_title} for course in courses]
        return JsonResponse(courses_data, safe=False)
    
    return JsonResponse({'error': 'Invalid department or semester'}, status=400)


def viewDetailofSpecficStudent(request, id):
    student = get_object_or_404(Student, user__id=id)
    courses = Course.objects.filter(department=student.department, semester=student.semester)
    
    selected_course = None
    attendance_records = None
    marks_records = None
    
    if request.method == "POST":
        selected_course_code = request.POST.get('courses')
        if selected_course_code:
            selected_course = get_object_or_404(Course, course_code=selected_course_code)
            attendance_records = Attendance.objects.filter(student=student, course=selected_course)
            marks_records = Marks.objects.filter(student=student, course=selected_course)
    
    context = {
        'student': student,
        'courses': courses,
        'selected_course': selected_course,
        'attendance_records': attendance_records,
        'marks_records': marks_records,
    }

    return render(request, 'student/student_detail.html', context)

def fetch_data(request):
    if request.method == 'POST':
        department_name = request.POST.get('departments')
        semester_no = request.POST.get('semester')
        course_code = request.POST.get('courses')

        print(department_name,semester_no,course_code)

        # Directly fetch attendance records based on provided inputs
        attendance_records = Attendance.objects.filter(
            course__course_code=course_code,
            course__department__Department_Name=department_name,
            semester__Semester_No=semester_no,
        )

        print('attendance_records  ',attendance_records)

        # Prepare the response data
        # response_data = []
        # for record in attendance_records:
        #     response_data.append({
        #         'student_name': f"{record.student.user.first_name} {record.student.user.last_name}",
        #         'student_roll_no': record.student.user.roll_no,
        #         'attendance_date': record.attendance_date,
        #         'is_present': record.is_present,
        #     })

        return render(request, 'admin/attendance/attendance_sheet.html', {'data': attendance_records ,'department_name':department_name ,'semester_no':semester_no,'course_code':course_code})

    else:

        return render(request, 'admin/attendance/attendance_sheet.html', {'error': 'Invalid request method'})

def searchByMonth(request):
    if request.method == 'POST':
        date = request.POST.get('date')
        if date:
            year, month = map(int, date.split('-'))
            attendances = Attendance.objects.filter(attendance_date__year=year, attendance_date__month=month)
        else:
            attendances = Attendance.objects.all()
        
        context = {
            'records': attendances,
            'search_date': date
        }
        return render(request, 'teacher/attendance_record.html', context)
    return render(request, 'teacher/attendance_record.html')

def count_attendance(request):
    attendance_data = Student.objects.all().annotate(
        total_present=Count('attendances', filter=Q(attendances__is_present=True)),
        total_absent=Count('attendances', filter=Q(attendances__is_present=False))
    )

    context = {
        'attendance_data': attendance_data,
    }

    return render(request, 'teacher/attendance_summary.html', context)


def searchByDate(request):
    if request.method == 'POST':
        date = request.POST.get('date')
        attendances = Attendance.objects.filter(attendance_date=date)
        context = {
            'records': attendances,
            'search_date': date
        }
        return render(request, 'teacher/attendance_record.html', context)
    return render(request, 'teacher/attendance_record.html')



def searchByDateforAdmin(request):
    if request.method == 'POST':
        date = request.POST.get('date')
        attendances = Attendance.objects.filter(attendance_date=date)
        print('attendances',attendances)
        context = {
            'data': attendances,
            'search_date': date
        }
        return render(request, 'admin/attendance/attendance_sheet.html', context)
    return render(request, 'admin/attendance/attendance_sheet.html')


def searchByDateforStudent(request):
    if request.method == 'POST':
        date = request.POST.get('date')
        cnic = request.POST.get('cnic')
        course_code = request.POST.get('course_code')
        semester_no = request.POST.get('semester_no')

        # Filtering attendance records based on the input criteria
        attendances = Attendance.objects.filter(
            attendance_date=date,
            student__user__cnic=cnic,
            course__course_code=course_code,
            semester__Semester_No=semester_no
        )

        # Prepare context data for rendering
        context = {
            'attendance_records': attendances,
            'search_date': date,
            'course_code': course_code,
            'semester_no': semester_no
        }

        return render(request, 'student/attendance_history.html', context)

    # If not a POST request, just render the empty template
    return render(request, 'student/attendance_history.html')
