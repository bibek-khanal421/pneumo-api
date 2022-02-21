from django.shortcuts import render
import requests
from .forms import ImageForm

# sending api request with the image file and get the response
def home(request):
    form=ImageForm()
    if request.method == 'POST':
        form=ImageForm(request.POST, request.FILES)
        image = request.FILES['image']
        url = 'http://localhost:8000/api/'
        files = {'inputImage': image}
        r = requests.post(url, files=files)
        print(r.text)
        return render(request,'Response.html', {'response': r.json()})
    return render(request, 'segmenter.html',{'form':form}) 


