# Generated by Django 4.1.1 on 2023-06-07 17:46

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Books',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(default='0', max_length=200)),
                ('book_id', models.CharField(default='0', max_length=200)),
                ('url', models.URLField(default='0')),
                ('cover_image', models.URLField(default='0')),
            ],
        ),
    ]
