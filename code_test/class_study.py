class Parent:
    def __init__(self, name):
        self.name = name
        print(f"Parent initialized with name: {self.name}")
    def hello(self):
        print("hello world!")
class Child(Parent):
    def __init__(self, name, age):
        super().__init__(name)  # 调用父类的初始化方法
        self.age = age
        print(f"Child initialized with age: {self.age}")
    def zileidef(self):
        self.hello()



# 实例化子类
child = Child("Alice", 12)
child.zileidef()
# 输出：
# Parent initialized with name: Alice
# Child initialized with age: 12
