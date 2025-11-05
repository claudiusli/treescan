Analyzer is a set of tools used to create a bitmask of an image that correspond to pixsels of a certain type. It is a single executable that changes its behavior based on the input arguments. JSON will be passed as strings eg "{"key1":"value1", "key2": val2}". As a first pass, if there are any errors, just catch them, report them, and exit.

Analyzer has several modes:
--normalize <string>
	opens <string> as an image
	it will then ouput the image as ppm p6 in the same directory as the input image but give it a .ppm extension it should maintain the same color depth as the original. 
	
	
--makesample
	takes a json object of the form:
	      {"image":<string>,
	      "x":<integer>,
	      "y":<integer>,
	      "w":<integer>,
	      "h":<integer>,
	      "color":<string>}
	      It will read the images file as a ppm or throw an error. If i tdoes not exist yet, it will create a directory named after the file with a .samples extension. It will create a subimages at those pixel coordintes and save it as a ppm in that subdirectory. the name of the sample will be of the format <x>_<y>_<w>_<h>.<color>
	      eg --makesample '{"image":"test.jpg","x":10,"y":20,"w":50,"h":50,"color":"blue"}'
	      these should all maintain the same color depth as the original. remember that the point of these samples is that any neural network trained on windows of the the samples should be directly applicable to the same size windows from the original.

--sampleselector
	takes a single string as an argument and interprets it as a ppm image
	displays that image on the screen (it may be large so disable decompression bomb protection)
	the user is then allowed to drag a selection box around any location on the screen. when they do so, a small popup appears with a space to enter a string, a cancel button and an OK button. The OK button is greyed out and inaccessible until some string is entered in the box. When the OK button is pressed a snippet is created from the selection box and it is saved as though makesample had been called with those parameters. Whether the user presses OK or cancel the program then continues and waits for the user to draw a new selection box. If the user closes the window, the program should exit.

--train
	takes a json object of the form:
	      {"samples":<string>,
	      "window":<integer>,
	      "traincount":<integer>}
	      It will open the <samples> directory and train a neural network on the files in that directory. It will assume that there are only 2 colors, as indicated by the extension of the files in that directory. It will ignore any model files in that directory. It will then do <traincount> training iterations. For each iteration it will randomly generate a square subimage from a randomly selected image of each color, and train the neural network on those samples. When it is done, it will output a model in that subdirectory with filename <size>.model If --model is specified, it will first load that as a model and modifey it, if not it will start with a random model of that size and iterate on it. The image is mostly differentated by color. the neural network should assume that the images will be ppd p6 formatted. the model files should ideally be human readable and contain metadata with a log of all the times it was updated and what files were used to update it as well as any options passed to analyze.py the specific format of JSON, YAML etc doesn't matter. There are multiple colors in the actual samples but they end up looking either mostly blue or mostly red. the random sampling should make sure that they provide full windows. Do not end trainign early but print the loss of the model about once per second (the exact time doesn't matter it just needs to be slow enough to read on a terminal). <traincount> will generally be much larger than the number of files so we can rely on the law of large numbers to get us good coverage.

--model takes a filename and loads a neural network of those dimensions with the weights. 

--test 
	takes a json object of the form:
	      {"samples":<string>,
	      "window":<integer>,
	      "testcount":<integer>}
	      It requires that --model is also specified.
	      It will open the <samples> directory and test samples against the supplied model. It will run <testcount> samples of each color, attempt to predict the color using the neural network and check if it was correct. At the end it should report the total number of each color that it actually analyzed and the number of each that it got correct. it should also express these as percentages. results should be printed on the command line.

--makemask
	takes a single argument "image":<string> as an argument.
	It requires that --model is also specified.
	This creates a ppm p6 bitmap that is <window>-1 pixels shorter and narrower than the size of <image>. that bitmap will be black if the the sample starting at the pixel would be analyzed as one color and white if would be analyzed as the other color. if the colors aren't black and white assign white to the one that comes first alphabetically and black to the other. the mask file should ge the name of the input file with "_mask" added before the extension.

This should use PIL, and pytorch.
collor assignment for the mask shouls use the colors from the filenames.
for --test and --makemask calculate the window size based on the model file.