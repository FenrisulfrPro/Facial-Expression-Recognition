from django.shortcuts import render


def link(request):
    return render(request, 'recognise.html')
