from django import forms
from django.contrib.auth.forms import SetPasswordForm
from django.forms import modelformset_factory

from .models import *


class DepartmentForm(forms.ModelForm):
    class Meta:
        model = Department
        fields = ['Department_Name', 'HeadOfDepartment']
        widgets = {
            'Department_Name': forms.TextInput(attrs={'class': 'form-control'}),
            'HeadOfDepartment': forms.TextInput(attrs={'class': 'form-control'}),
        }
        labels = {
            'Department_Name': 'Department Name',
            'HeadOfDepartment': 'Head of Department',
        }

class SemesterForm(forms.ModelForm):
    class Meta:
        model = Semester
        fields = ['Semester_No', 'DepartmentName']
        labels = {
            'Semester_No': 'Semester Number',
            'DepartmentName': 'Department Name',
        }
        widgets = {
            'Semester_No': forms.NumberInput(attrs={'class': 'form-control'}),
            'DepartmentName': forms.Select(attrs={'class': 'form-control'}),
        }

class CourseForm(forms.ModelForm):
    class Meta:
        model = Course
        fields = ['course_code', 'course_title', 'semester', 'department']
        labels = {
            'course_code': 'Course Code',
            'course_title': 'Course Title',
            'semester': 'Semester',
            'department': 'Department',
        }
        widgets = {
            'course_code': forms.TextInput(attrs={'class': 'form-control'}),
            'course_title': forms.TextInput(attrs={'class': 'form-control'}),
            'semester': forms.Select(attrs={'class': 'form-control'}),
            'department': forms.Select(attrs={'class': 'form-control'}),
        }

class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}), label='Password')

    class Meta:
        model = User
        fields = ['cnic', 'registration_no', 'roll_no', 'email', 'password', 'phone_no', 'gender', 'image', 'age', 'address']

        widgets = {
            'cnic': forms.TextInput(attrs={'class': 'form-control'}),
            'registration_no': forms.TextInput(attrs={'class': 'form-control'}),
            'roll_no': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'phone_no': forms.TextInput(attrs={'class': 'form-control'}),
            'gender': forms.Select(attrs={'class': 'form-control'}),
            'image': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'age': forms.NumberInput(attrs={'class': 'form-control'}),
            'address': forms.TextInput(attrs={'class': 'form-control'}),
        }
        labels = {
            'cnic': 'CNIC',
            'registration_no': 'Registration Number',
            'roll_no': 'Roll Number',
            'email': 'Email',
            'password': 'Password',
            'phone_no': 'Phone Number',
            'gender': 'Gender',
            'image': 'Profile Image',
            'age': 'Age',
            'address': 'Address',
        }

class UserLoginForm(forms.Form):
    cnic = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control'}), label='CNIC')
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}), label='Password')

class StudentForm(forms.ModelForm):
    cnic = forms.CharField(max_length=20, widget=forms.TextInput(attrs={'class': 'form-control'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}), label='Password')
    registration_no = forms.CharField(max_length=25, widget=forms.TextInput(attrs={'class': 'form-control'}))
    roll_no = forms.CharField(max_length=25, widget=forms.TextInput(attrs={'class': 'form-control'}))
    first_name = forms.CharField(max_length=25, widget=forms.TextInput(attrs={'class': 'form-control'}))
    email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control'}))
    phone_no = forms.CharField(max_length=15, widget=forms.TextInput(attrs={'class': 'form-control'}))
    gender = forms.ChoiceField(choices=User.GENDER_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    image = forms.ImageField(required=False, widget=forms.FileInput(attrs={'class': 'form-control-file'}))
    age = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    address = forms.CharField(max_length=150, widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3}))
    department = forms.ModelChoiceField(queryset=Department.objects.all(), widget=forms.Select(attrs={'class': 'form-control'}))
    semester = forms.ModelChoiceField(queryset=Semester.objects.all(), widget=forms.Select(attrs={'class': 'form-control'}))

    class Meta:
        model = Student
        fields = ['cnic','password', 'registration_no', 'roll_no','first_name', 'email', 'phone_no','gender', 'image', 'age', 'address', 'department','semester']

class ParentForm(forms.ModelForm):
    father__cnic = forms.CharField(max_length=20, widget=forms.TextInput(attrs={'class': 'form-control'}))
    father__password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}), label='Password')
    father__first_name = forms.CharField(max_length=50, widget=forms.TextInput(attrs={'class': 'form-control'}))
    father__email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control'}))
    father__phone_no = forms.CharField(max_length=15, widget=forms.TextInput(attrs={'class': 'form-control'}))

    class Meta:
        model = Parent
        fields = ['father__cnic','father__password','father__first_name', 'father__email', 'father__phone_no']

class TeacherForm(forms.ModelForm):
    teacher__cnic = forms.CharField(max_length=20, label='CNIC', widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter CNIC'}))
    teacher__password = forms.CharField(label='Password', widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Enter Password'}))
    teacher__first_name = forms.CharField(max_length=50, label='First Name', widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter First Name'}))
    teacher__email = forms.EmailField(label='Email', widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Enter Email'}))
    teacher__phone_no = forms.CharField(max_length=15, label='Phone Number', widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Phone Number'}))
    department = forms.ModelChoiceField(queryset=Department.objects.all(), label='Department', empty_label=None, widget=forms.Select(attrs={'class': 'form-control'}))
   
    class Meta:
        model = Teacher
        fields = ['teacher__cnic', 'teacher__password', 'teacher__first_name', 'teacher__email', 'teacher__phone_no', 'department']
        labels = {
            'teacher__cnic': 'CNIC',
            'teacher__password': 'Password',
            'teacher__first_name': 'First Name',
            'teacher__email': 'Email',
            'teacher__phone_no': 'Phone Number',
            'department': 'Department',
        }
        widgets = {
            'teacher__cnic': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter CNIC'}),
            'teacher__password': forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Enter Password'}),
            'teacher__first_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter First Name'}),
            'teacher__email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Enter Email'}),
            'teacher__phone_no': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Phone Number'}),
            'department': forms.Select(attrs={'class': 'form-control'}),
        }

class AssignmentForm(forms.ModelForm):
    class Meta:
        model = Assignment
        fields = ['teacher', 'course', 'semester']


class AttendanceSelectionForm(forms.Form):
    department = forms.ModelChoiceField(queryset=Department.objects.all(), required=True, label='Department')
    semester = forms.ModelChoiceField(queryset=Semester.objects.all(), required=True, label='Semester')
    course = forms.ModelChoiceField(queryset=Course.objects.all(), required=True, label='Course')


class UserUpdateForm(forms.ModelForm):
    class Meta:
        model = User
        fields = [
            'cnic', 'registration_no', 'roll_no', 'first_name', 'password',
            'last_name', 'email', 'phone_no', 'gender', 
            'address', 'age'
        ]

class AdminPasswordChangeForm(SetPasswordForm):
    class Meta:
        model = User
        fields = ['password']


class AdminPasswordResetForm(forms.Form):
    cnic = forms.CharField(max_length=20, label='CNIC')
    new_password = forms.CharField(widget=forms.PasswordInput, label='New Password')


class TimeTableForm(forms.ModelForm):
    class Meta:
        model = TimeTable
        fields = ['title', 'image']



class MarksForm(forms.ModelForm):
    class Meta:
        model = Marks
        fields = ['marks_obtained', 'total_marks']

MarksFormSet = modelformset_factory(
    Marks,
    form=MarksForm,
    extra=0,
    can_delete=False,
)