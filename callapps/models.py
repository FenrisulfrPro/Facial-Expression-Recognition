from django.db import models


# Create your models here.


class Admin(models.Model):
    """管理员表"""
    username = models.CharField(verbose_name='用户名', max_length=32)
    password = models.CharField(verbose_name='密码', max_length=64)

    def __str__(self):
        return self.username


class Department(models.Model):
    """部门表"""
    title = models.CharField(verbose_name='标题', max_length=32)

    def __str__(self):
        return self.title


class UserInfo(models.Model):
    """来访者表"""
    name = models.CharField(verbose_name='姓名', max_length=16)
    password = models.CharField(verbose_name='密码', max_length=64)
    age = models.IntegerField(verbose_name='年龄')
    account = models.DecimalField(verbose_name='账户积分', max_digits=10, decimal_places=2, default=0)
    # create_time = models.DateTimeField(verbose_name='来访时间')
    create_time = models.DateField(verbose_name='来访时间')

    # 无约束
    # depart_id = models.BigIntegerField(verbose_name='部门id')

    # 1.有约束
    #   -to,与那张表关联
    #   -to_field,表中的那一列关联
    # 2.Django自动
    #   -写的depart
    #   -生成数据列 depart_id
    # 3.部门表被删除
    #  3.1 级联删除  on_delete=models.CASCADE  表示如果部门表被删除，旗下的部门表的来访者也被删除
    depart = models.ForeignKey(verbose_name='部门', to='Department', to_field='id', on_delete=models.CASCADE)
    #  3.2 置空  null=True, blank=True, on_delete=models.SET_NULL 表示如果部门表被删除，旗下的部门表的来访者部门位为空值
    # depart = models.ForeignKey(to='Department', to_field='id', null=True, blank=True, on_delete=models.SET_NULL)

    # 在Django中做的约束
    gender_choices = (
        (1, '男'),
        (2, '女'),
    )
    gender = models.SmallIntegerField(verbose_name='性别', choices=gender_choices)

    # def __str__(self):
    #     return self.name
