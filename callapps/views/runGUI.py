from django.shortcuts import render
import os


def run(request):
    os.system('python GUI.py')
    return render(request, 'call.html', {'text': '表情识别结束，感谢您的使用'})
