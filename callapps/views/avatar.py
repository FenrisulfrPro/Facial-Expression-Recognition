from django.shortcuts import render, redirect
"""
更改头像 视图函数
"""


def avatar_edit(request):
    """更改头像"""
    if request.method == 'GET':
        return render(request, 'avatar_edit.html')

    file = request.FILES.get('avatar')
    file_name = 'media/avatar/'+file.name

    with open(file_name, 'wb+') as fo:
        for chunk in file.chunks():
            fo.write(chunk)

    # 将用户头像写到session
    info = request.session.get('info')
    request.session['info'] = {
        'id': info['id'],
        'name': info['name'],
        'avatar': '/'+file_name,
    }

    return redirect('/recognise/')