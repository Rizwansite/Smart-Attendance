# Generated by Django 4.2.7 on 2024-04-30 11:48

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0016_student_teachers'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='student',
            name='teachers',
        ),
    ]
