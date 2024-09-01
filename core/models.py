from django.contrib.auth.models import AbstractUser
from django.db import models
from .manager import UserManager

class Department(models.Model):
    Department_Name = models.CharField(max_length=100, primary_key=True,null=False,blank=False)
    HeadOfDepartment = models.CharField(max_length=100)
    
    def __str__(self):
        return self.Department_Name


class Semester(models.Model):
    Semester_No = models.CharField(max_length=4,default=1)
    DepartmentName = models.ForeignKey(Department, on_delete=models.CASCADE, related_name='semesters')

    class Meta:
        unique_together = ('Semester_No', 'DepartmentName')


    def __str__(self):
        return f"Semester {self.Semester_No} - {self.DepartmentName.Department_Name}"

class Course(models.Model):
    course_code = models.CharField(max_length=25, primary_key=True)
    course_title = models.CharField(max_length=100)
    semester = models.ForeignKey(Semester, on_delete=models.CASCADE, related_name="courses")
    department = models.ForeignKey(Department, on_delete=models.CASCADE, related_name="courses")

    def __str__(self):
        return f"{self.course_code} -- {self.course_title}"


class User(AbstractUser):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
    ]

    username = None
    # id = models.AutoField(primary_key=True,default=1)
    cnic = models.CharField(max_length=20,unique=True)
    registration_no = models.CharField(max_length=25, unique=True ,null=True,blank=True)
    roll_no = models.CharField(max_length=25, unique=True ,null=True,blank=True)
    email = models.EmailField(unique=True)
    phone_no = models.CharField(max_length=15)
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    image = models.ImageField(upload_to='student_images/', blank=True, null=True)
    date = models.DateField(auto_now_add=True)
    last_login = models.DateTimeField(blank=True, null=True)
    logout_date = models.DateTimeField(blank=True, null=True)
    age = models.PositiveIntegerField(null=True)
    address = models.CharField(max_length=150)


    objects = UserManager()

    USERNAME_FIELD = 'cnic'
    REQUIRED_FIELDS = ['email']

    def _str_(self):
        return self.cnic


class Parent(models.Model):

     user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)   
    
     def __str__(self):
         return self.user.cnic

class Student(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    department = models.ForeignKey(Department, on_delete=models.CASCADE, default='')
    parent = models.ForeignKey(Parent, on_delete=models.CASCADE, blank=True, null=True)
    semester = models.ForeignKey(Semester, on_delete=models.CASCADE, default=3)

    def __str__(self):
        return self.user.cnic

class Teacher(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    department = models.ForeignKey(Department, on_delete=models.CASCADE, default='')

    def __str__(self):
        return f"{self.user.first_name} -- {self.department.Department_Name}"

class Assignment(models.Model):
    teacher = models.ForeignKey(Teacher, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    semester = models.ForeignKey(Semester, on_delete=models.CASCADE)  

class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE,related_name='attendances')
    attendance_date = models.DateField(auto_now_add=True)
    is_present = models.BooleanField(default=False, verbose_name='Is Present')
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    semester = models.ForeignKey(Semester, on_delete=models.CASCADE)


    def __str__(self):
        return f"Attendance: {self.student.user.first_name} {self.student.user.last_name} - {self.course.course_title} on {self.attendance_date}"

class TimeTable(models.Model):
    title = models.CharField(max_length=100)
    image = models.ImageField(upload_to='timetable_images/')

    def __str__(self):
        return self.title


class Marks(models.Model):
    teacher = models.ForeignKey(Teacher, on_delete=models.CASCADE)
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    semester = models.ForeignKey(Semester, on_delete=models.CASCADE)
    marks_obtained = models.DecimalField(max_digits=5, decimal_places=2)
    total_marks = models.DecimalField(max_digits=5, decimal_places=2)
    date_uploaded = models.DateField(auto_now_add=True)

    class Meta:
        unique_together = ('student', 'course', 'semester')

    def __str__(self):
        return f"Marks: {self.student.user.cnic} - {self.course.course_title} - {self.marks_obtained}/{self.total_marks}"
    
class CourseAttendanceCount(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    semester = models.ForeignKey(Semester, on_delete=models.CASCADE)
    count = models.PositiveIntegerField(default=0)

    class Meta:
        unique_together = ('course', 'semester')

    def __str__(self):
        return f"{self.count} "


# class savedimage(models.Model):
#     imagee = models.ImageField(upload_to='attendance_Images/', blank=True, null=True)
#     pass
