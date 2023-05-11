"""
来访者 视图函数
"""
from django.shortcuts import render, redirect
from callapps import models
from callapps.utils.pagination import Pagination  # 自定义的分页组件
from callapps.utils.form import UserModelForm


def user_list(request):
    """来访者列表"""
    # 构造搜索条件
    # data_dict = {}
    # search_data = request.GET.get('q', '')
    # if search_data:
    #     data_dict['name__contains'] = search_data

    # 根据搜索条件去数据库获取
    # queryset = models.Admin.objects.filter(**data_dict)

    # 获取所有来访者列表 [obj,obj,obj,]
    queryset = models.UserInfo.objects.all()

    # 分页
    page_object = Pagination(request, queryset)

    context = {
        'queryset': page_object.page_queryset,
        'page_string': page_object.html(),
        # 'search_data': search_data,
    }
    return render(request, 'user_list.html', context)


def user_add(request):
    """添加来访者（ModelForm版本）"""
    title = '添加来访者'
    if request.method == "GET":
        form = UserModelForm
        return render(request, 'user_change.html', {"form": form, 'title': title})

    # 用户POST提交数据，数据校验
    form = UserModelForm(data=request.POST)
    if form.is_valid():
        # 如果数据合法，保存到数据库
        form.save()
        return redirect('/user/list/')

    # 校验失败，在页面上显示错误信息
    return render(request, 'user_change.html', {"form": form, 'title': title})


def user_edit(request, nid):
    """编辑来访者"""
    title = '编辑来访者'
    # 根据id去数据库获取要编辑的那一行数据（对象）
    row_object = models.UserInfo.objects.filter(id=nid).first()

    if request.method == "GET":
        form = UserModelForm(instance=row_object)
        return render(request, 'user_change.html', {"form": form, 'title': title})

    form = UserModelForm(data=request.POST, instance=row_object)
    if form.is_valid():
        # 默认保存的是来访者输入的所有数据，如果想要再来访者输入以外增加一点值
        # form.instance.字段名 = 值
        form.save()
        return redirect('/user/list')
    return render(request, 'user_change.html', {"form": form, 'title': title})


def user_delete(request, nid):
    """删除来访者"""
    models.UserInfo.objects.filter(id=nid).delete()
    return redirect('/user/list')
