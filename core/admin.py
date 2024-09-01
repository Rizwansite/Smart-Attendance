from django.contrib import admin
from .models import *


admin.site.register(User)
admin.site.register(Department)
admin.site.register(Semester)
admin.site.register(Student)
admin.site.register(Parent)
admin.site.register(Course)
admin.site.register(Teacher)
admin.site.register(Attendance)
admin.site.register(Assignment)
admin.site.register(Marks)