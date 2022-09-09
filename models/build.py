class BuildModel:
    buildFunc = {}
    def __init__(self):
        print("Model Build prepared...\n")

    @classmethod
    def build(BuildModel,name:int):
        model = BuildModel.buildFunc[name]()

        print("Model Build %d :%s \n"%(name,getattr(BuildModel.buildFunc[name],'__name__')))
        return model

    @classmethod
    def add(BuildModel,name:int,function):

        BuildModel.buildFunc[name] = function
        print("Model Add %d : %s \n"%(name,getattr(function,'__name__')))

if __name__ == "__main__":
    BuildModel.add(BuildModel,"w","we")
    print(BuildModel.build("w"))
