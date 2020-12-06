# OCR reading local text

![](https://pandao.github.io/editor.md/images/logos/editormd-logo-180x180.png)



### How to use this code

- First, you have to install tesseract on your machine. https://medium.com/quantrium-tech/installing-and-using-tesseract-4-on-windows-10-4f7930313f82
- All of the setting in this code is in .json file. you can check it out in config.
- all default is set you can test demo by run main.py

### Algorithm
- we use EAST text detector to crop text boxes. 
- read text in those text box and show the result in the boxes. 
