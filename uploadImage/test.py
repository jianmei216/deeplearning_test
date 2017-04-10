import pycurl
from io import StringIO
from io import BytesIO
b = BytesIO();
def write(bytes):
    print(bytes)

c = pycurl.Curl()
c.setopt(c.HEADERFUNCTION, b.write)
c.setopt(c.POST, 1)
c.setopt(c.URL, "http://image.baidu.com/pictureup/uploadshitu")
c.setopt(c.HTTPPOST, [("image", (c.FORM_FILE, "files/invoice.jpg"))])
c.perform()
print(b.getvalue().decode('utf-8'))
b.close()
c.close()
print( "that's it")