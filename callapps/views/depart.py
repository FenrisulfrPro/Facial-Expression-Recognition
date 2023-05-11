"""
学院 视图函数
"""
from django.shortcuts import render, redirect
from callapps import models
from callapps.utils.pagination import Pagination  # 自定义的分页组件
from callapps.utils.form import DepartModelForm


def depart_list(request):
    """部门列表"""
    # 构造搜索条件
    # data_dict = {}
    # search_data = request.GET.get('q', '')
    # if search_data:
    #     data_dict['title__contains'] = search_data

    # 根据搜索条件去数据库获取
    # queryset = models.Admin.objects.filter(**data_dict)

    # 获取所有部门列表 [obj,obj,obj,]
    queryset = models.Department.objects.all()

    # 分页
    page_object = Pagination(request, queryset)

    context = {
        'queryset': page_object.page_queryset,
        'page_string': page_object.html(),
        # 'search_data': search_data,
    }

    return render(request, 'depart_list.html', context)


def depart_add(request):
    """添加部门"""
    title = '添加部门'
    if request.method == 'GET':
        form = DepartModelForm
        return render(request, 'change.html', {"form": form, 'title': title})

    # 用户POST提交数据，数据校验
    form = DepartModelForm(data=request.POST)
    if form.is_valid():
        # 如果数据合法，保存到数据库
        form.save()
        return redirect('/depart/list/')

    # 校验失败，在页面上显示错误信息
    return render(request, 'change.html', {"form": form, 'title': title})


def depart_delete(request, nid):
    """删除部门"""
    models.Department.objects.filter(id=nid).delete()
    return redirect('/depart/list/')


def depart_edit(request, nid):
    """修改部门"""
    title = '修改部门'
    # 根据id去数据库获取要编辑的那一行数据（对象）
    row_object = models.Department.objects.filter(id=nid).first()

    if request.method == 'GET':
        form = DepartModelForm(instance=row_object)
        return render(request, 'change.html', {"form": form, 'title': title})

    form = DepartModelForm(data=request.POST, instance=row_object)
    if form.is_valid():
        # 校验成功
        form.save()
        return redirect('/depart/list/')

    # 校验失败，提示错误信息
    return render(request, 'change.html', {"form": form, 'title': title})
