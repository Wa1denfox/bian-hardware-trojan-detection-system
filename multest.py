from test import *
def multytest(model_path):
    for file in pdfFilesPath('test'):
        testfunc(file,model_path)
if __name__ == '__main__':
    print("======================开始检测======================")
    multytest('model_show.pth')
    print("======================检测结束======================")
