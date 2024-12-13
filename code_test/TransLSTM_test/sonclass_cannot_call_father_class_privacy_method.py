class Parent:
    def __private_method(self):
        print("This is a private method of Parent.")

class Child(Parent):
    def access_private(self):
        self._Parent__private_method()  # 使用重整后的名称调用私有方法

child = Child()
child.access_private()
