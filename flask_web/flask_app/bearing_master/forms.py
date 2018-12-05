#!/usr/bin/python
#-*- coding: UTF-8 -*-
from __future__ import unicode_literals
from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField, StringField, PasswordField, SelectField, BooleanField, FileField
from wtforms.validators import DataRequired, Length

class TodoListForm(FlaskForm):
    title = StringField('标题', validators=[DataRequired(), Length(1, 64)])
    status = RadioField('是否完成', validators=[DataRequired()],  choices=[("0", '机器学习模型'),("1",'深度神经网络')])
    submit = SubmitField('提交')

class DataListForm(FlaskForm):
    title = StringField('标题', validators=[DataRequired(), Length(1, 64)])
    submit = SubmitField('提交')

class LoginForm(FlaskForm):
    username = StringField('用户名', validators=[DataRequired(), Length(1, 24)])
    password = PasswordField('密码', validators=[DataRequired(), Length(1, 24)])
    submit = SubmitField('登录')


class RegistorForm(FlaskForm):
    username = StringField('用户名', validators=[DataRequired(), Length(1, 24)])
    password = PasswordField('密码', validators=[DataRequired(), Length(1, 24)])
    submit = SubmitField('注册')


class CNNForm(FlaskForm):
    # submit = SubmitField("Submit")
    # username = StringField('网络类型名', validators=['cnn', Length(0, 0)])
    submit = SubmitField('深度学习')
class MLForm(FlaskForm):
    # submit = SubmitField("Submit")
    # username = StringField('网络类型名', validators=['cnn', Length(0, 0)])
    submit = SubmitField('机器学习')




class CNNSetting(FlaskForm):
    """电影表单"""

    tag_id00 = SelectField(
        label="网络类型",
        validators=[DataRequired("请选择标签")],
        description="",
        coerce=int,
        render_kw={"class": "form-control"}
    )
    tag_id01 = SelectField(
        label="激活函数",
        validators=[DataRequired("请选择标签")],
        description="",
        coerce=int,
        render_kw={"class": "form-control"}
    )
    tag_id02 = SelectField(
        label="优化器",
        validators=[DataRequired("请选择标签")],
        description="",
        coerce=int,
        render_kw={"class": "form-control"}
    )

    tag_id03 = SelectField(
        label="batch size",
        validators=[DataRequired("请选择标签")],
        description="",
        coerce=int,
        render_kw={"class": "form-control"}
    )




    tag_id013 = StringField('输入维度', validators=[DataRequired(), Length(1, 24)])
    tag_id014 = StringField('输出维度', validators=[DataRequired(), Length(1, 24)])
    tag_id015 = StringField('网络层数', validators=[DataRequired(), Length(1, 24)])
    tag_id016 = StringField('学习率', validators=[DataRequired(), Length(1, 24)])
    tag_id017 = StringField('权值衰减率', validators=[DataRequired(), Length(1, 24)])


    # tag_id18 = BooleanField('是否选择', validators=[DataRequired()])
    # tag_id19 = BooleanField('是否选择2', validators=[DataRequired()])


    # submit = SubmitField('确定')

    # star = SelectField(
    #         label="星级",
    #         validators=[DataRequired("请选择星际")],
    #         description="星级",
    #         coerce=int,
    #         choices=[(1, "1星"), (2, "2星"), (3, "3星"), (4, "4星"), (5, "5星")],
    #         render_kw={"class": "form-control"})

    # def __init__(self, *args, **kwargs):
    #     super(CNNSetting, self).__init__(*args, **kwargs)
    #     self.tag_id.choices = [(v.id, v.name) for v in Tag.query.all()]
    def __init__(self, args):
        super(CNNSetting, self).__init__(args)
        # super(CNNSetting, self).__init__()
        print(args)
        self.tag_id00.choices = [(i, args[0][i]) for i in range(len(args[0]))]
        self.tag_id01.choices = [(i, args[1][i]) for i in range(len(args[1]))]
        self.tag_id02.choices = [(i, args[2][i]) for i in range(len(args[2]))]
        self.tag_id03.choices = [(i, args[3][i]) for i in range(len(args[3]))]



class Data_Select_Form():
    # tag_id18 = BooleanField('是否选择1', validators=[DataRequired()])
    # tag_id19 = BooleanField('是否选择1', validators=[DataRequired()])
    # tag_id = []
    # for i in range(100):
    #     tag_id.append(BooleanField('是否选择', validators=[DataRequired()]))

    def __init__(self, args):
        print(args)

        class DynamicForm(FlaskForm):

            #############
            #############
            def mygetForm(self,str_name):
                return self.__getattribute__(str_name)
            pass
        for i in range(args[0]):
            # setattr(DynamicForm, 'tag_id' + str(i), StringField(i))
            # setattr(DynamicForm, 'tag_id' + str(i), BooleanField(str(i)))
            setattr(DynamicForm, 'tag_id' + str(i), BooleanField(str(i), validators=[DataRequired()]))
        setattr(DynamicForm, 'submit', SubmitField('确定'))

        self.form = DynamicForm()

    def __call__(self):
        return self.form
            # for i in range(args[0]):
        #     self.tag_id.append(BooleanField('是否选择', validators=[DataRequired()]))


class All_Set_Form():
    # tag_id18 = BooleanField('是否选择1', validators=[DataRequired()])
    # tag_id19 = BooleanField('是否选择1', validators=[DataRequired()])
    # tag_id = []
    # for i in range(100):
    #     tag_id.append(BooleanField('是否选择', validators=[DataRequired()]))
    def __init__(self, args):
        print(args)

        class DynamicForm(FlaskForm):
            tag_id00 = SelectField(
                label="网络类型",
                validators=[DataRequired("请选择标签")],
                description="",
                coerce=int,
                render_kw={"class": "form-control"}
            )
            tag_id01 = SelectField(
                label="激活函数",
                validators=[DataRequired("请选择标签")],
                description="",
                coerce=int,
                render_kw={"class": "form-control"}
            )
            tag_id02 = SelectField(
                label="优化器",
                validators=[DataRequired("请选择标签")],
                description="",
                coerce=int,
                render_kw={"class": "form-control"}
            )

            tag_id03 = SelectField(
                label="batch size",
                validators=[DataRequired("请选择标签")],
                description="",
                coerce=int,
                render_kw={"class": "form-control"}
            )

            tag_id04 = SelectField(
                label="是否做数据增强",
                validators=[DataRequired("请选择标签")],
                description="",
                coerce=int,
                render_kw={"class": "form-control"}
            )
            tag_id05 = SelectField(
                label="损失函数",
                validators=[DataRequired("请选择标签")],
                description="",
                coerce=int,
                render_kw={"class": "form-control"}
            )


            tag_id013 = StringField('输入维度', validators=[DataRequired(), Length(1, 24)])
            tag_id014 = StringField('输出维度', validators=[DataRequired(), Length(1, 24)])
            tag_id015 = StringField('网络层数', validators=[DataRequired(), Length(1, 24)])
            tag_id016 = StringField('学习率', validators=[DataRequired(), Length(1, 24)])
            tag_id017 = StringField('权值衰减率', validators=[DataRequired(), Length(1, 24)])

            # whether_data_augment =, deep_model_class =, ml_model_class = -1, input_dim =, output_dim =, weight_decay =,
            # learning_rate =, activation_class =, layers_num =

            def __init__(self, args):
                # super(DynamicForm, self).__init__(args)
                super(DynamicForm, self).__init__()
                print(args)
                # self.tag_id00.choices = [(1, v[1]) for v in args]
                # self.tag_id01.choices = [(1, v[1]) for v in args]
                # self.tag_id02.choices = [(1, v[1]) for v in args]
                # self.tag_id03.choices = [(1, v[1]) for v in args]

                self.tag_id00.choices = [(i, args[0][i]) for i in range(len(args[0]))]
                self.tag_id01.choices = [(i, args[1][i]) for i in range(len(args[1]))]
                self.tag_id02.choices = [(i, args[2][i]) for i in range(len(args[2]))]
                self.tag_id03.choices = [(i, args[3][i]) for i in range(len(args[3]))]
                self.tag_id04.choices = [(i, args[4][i]) for i in range(len(args[4]))]
                self.tag_id05.choices = [(i, args[5][i]) for i in range(len(args[5]))]

            #############
            #############
            def mygetForm(self,str_name):
                return self.__getattribute__(str_name)
            pass
        for i in range(args[0]):
            # setattr(DynamicForm, 'tag_id' + str(i), StringField(i))
            # setattr(DynamicForm, 'tag_id_dym' + str(i), BooleanField(str(i), validators=[DataRequired()]))
            setattr(DynamicForm, 'tag_id_dym' + str(i), BooleanField(str(i), validators=[]))
        setattr(DynamicForm, 'submit', SubmitField('确定'))

        self.form = DynamicForm(args[1])

    def __call__(self):
        return self.form




class ML_Set_Form():
    def __init__(self, args):
        print('arg out:',args)
        class DynamicForm(FlaskForm):
            tag_id00 = SelectField(
                label="网络类型",
                validators=[DataRequired("请选择标签")],
                description="",
                coerce=int,
                render_kw={"class": "form-control"}
            )
            tag_id01 = SelectField(
                label="激活函数",
                validators=[DataRequired("请选择标签")],
                description="",
                coerce=int,
                render_kw={"class": "form-control"}
            )
            tag_id013 = StringField('输入维度', validators=[DataRequired(), Length(1, 24)])
            tag_id014 = StringField('输出维度', validators=[DataRequired(), Length(1, 24)])
            tag_id016 = StringField('学习率', validators=[DataRequired(), Length(1, 24)])
            tag_id017 = StringField('权值衰减率', validators=[DataRequired(), Length(1, 24)])

            def __init__(self, args):
                super(DynamicForm, self).__init__(args)
                # super(DynamicForm, self).__init__()
                print('arg',args)
                self.tag_id00.choices = [(i, args[0][i]) for i in range(len(args[0]))]
                self.tag_id01.choices = [(i, args[1][i]) for i in range(len(args[1]))]
            #############
            def mygetForm(self,str_name):
                return self.__getattribute__(str_name)
            pass
        for i in range(args[0]):
            # setattr(DynamicForm, 'tag_id' + str(i), StringField(i))
            # setattr(DynamicForm, 'tag_id_dym' + str(i), BooleanField(str(i), validators=[DataRequired()]))
            setattr(DynamicForm, 'tag_id_dym' + str(i), BooleanField(str(i), validators=[]))
        setattr(DynamicForm, 'submit', SubmitField('确定'))
        self.form = DynamicForm(args[1])
    def __call__(self):
        return self.form



class DataSetting(FlaskForm):
    """电影表单"""
    tag_id = SelectField(
        label="输入维度",
        validators=[DataRequired("请选择标签")],
        description="",
        coerce=int,
        render_kw={"class": "form-control"}
    )

    tag_id13 = StringField('均值', validators=[DataRequired(), Length(1, 24)])

    # submit = SubmitField('确定')



    # star = SelectField(
    #         label="星级",
    #         validators=[DataRequired("请选择星际")],
    #         description="星级",
    #         coerce=int,
    #         choices=[(1, "1星"), (2, "2星"), (3, "3星"), (4, "4星"), (5, "5星")],
    #         render_kw={"class": "form-control"})

    # def __init__(self, *args, **kwargs):
    #     super(CNNSetting, self).__init__(*args, **kwargs)
    #     self.tag_id.choices = [(v.id, v.name) for v in Tag.query.all()]
    def __init__(self, args):
        # super(DataSetting, self).__init__(args)
        super(DataSetting, self).__init__()
        print(args)
        self.tag_id.choices = [(v[0], v[1]) for v in args]


class UploadForm(FlaskForm):
    image = FileField('选择数据')
    submit = SubmitField('提交')

