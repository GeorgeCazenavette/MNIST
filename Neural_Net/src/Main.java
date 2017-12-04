/*

George Cazenavette
102-43-135
Assignment 2: MNIST Handwritten Digit Recognizer Neural Network in Java
Due: 10/16/2017

This program reads in a training set of 50000 hand-written digits
and uses Stochastic Gradient Descent learning to develop a neural network
capable of correctly identifying handwritten digits in a disjoint set of
10000 testing images.

The program can also save a generated set of weights and biases to be
loaded in the future.

***IMPORTANT***
You may need to allocate more heap space for the JVM in order for this program to function properly

*/

import java.io.*;
import java.util.*;

class Main {

    // These are the names of the files used by the program.
    // If they are different, they should be changed here.
    private final static String TRAINING_DATA =     "mnist_train.csv";
    private final static String TESTING_DATA =      "mnist_test.csv";
    private final static String SAVED_NET =         "weights.csv";
    
    // Other parameters that may be edited by the user
    private final static int TRAINING_IMAGES =  50000;
    private final static int TESTING_IMAGES =   10000;
    private final static int        EPOCHS =    30;
    private final static int    BATCH_SIZE =    10;
    private final static double         ETA =   3;
    private final static int[] LAYER_SIZES =    {784, 100, 10};

    // main method
    public static void main(String[] args)
    {
        // Creates a new front end object from which the main program is run.
        FrontEnd app = new FrontEnd(TRAINING_DATA, TESTING_DATA, SAVED_NET, TRAINING_IMAGES, TESTING_IMAGES, 
                EPOCHS, BATCH_SIZE, ETA, LAYER_SIZES);
        app.run();
    }
}

// handles all interaction with the user
class FrontEnd
{

    private boolean hasWeights; // This variable determines if specific options are shown

    // Constant values passed in from the main method
    private final String TRAINING_FILE;
    private final String TESTING_FILE;
    private final String WEIGHTS_FILE;
    private final int TESTING_IMAGES;
    private final int TRAINING_IMAGES;
    private final int EPOCHS;
    private final int BATCH_SIZE;
    private final double ETA;

    // Initializes the neural net
    private final Network neuralNet;

    // FrontEnd constructor.
    // Initialises the file names and creates the neural network object
    // Initializes constants
    public FrontEnd(String trainingFile, String testingFile, String weightsFile, int trainingImages, int testingImages,
                    int epochs, int batchSize, double eta, int[] sizes)
    {
        hasWeights = false;

        neuralNet = new Network(sizes);

        TRAINING_FILE = trainingFile;
        TESTING_FILE = testingFile;
        WEIGHTS_FILE = weightsFile;
        TRAINING_IMAGES = trainingImages;
        TESTING_IMAGES = testingImages;
        EPOCHS = epochs;
        BATCH_SIZE = batchSize;
        ETA = eta;
    }

    // Prints the options available to the user and waits for input
    // Some options are only available if a network has already been generated or loaded
    private String menu()
    {
        clearScreen();
        System.out.println("[1] Train the network.");
        System.out.println("[2] Load a pre-trained network.");
        if (this.hasWeights)
        {
            System.out.println("[3] Display network accuracy on TRAINING data.");
            System.out.println("[4] Display network accuracy on TESTING data.");
            System.out.println("[5] Save the network state to file.");
            System.out.println("[6] Show TESTING data graphically.");
            System.out.println("[7] Show only incorrectly classified TESTING data graphically.");
        }
        System.out.println("[0] Exit");

        // gets user input and returns it
        return getInput();
    }

    // Executes a command based on the user input
    // Many commands will only execute if a set of weights and biases has been generated or loaded
    // Each case corresponds to the same main menu option
    private void processInput(String input)
    {
        switch (input)
        {
            case "1":   // Executes Stochastic Gradient Descent to build weights and biases
                System.out.println("\nTraining network...\n");
                neuralNet.SGD(EPOCHS, BATCH_SIZE, ETA, TRAINING_FILE, TRAINING_IMAGES);
                hasWeights = true;  // training generates a set of weights and biases
                break;
            case "2":   // Loads a previously saved network
                System.out.println("\nAttempting to load network...\n");
                if (neuralNet.load(WEIGHTS_FILE))
                {
                    hasWeights = true;  // after loading, there will be a set of weights and biases
                    System.out.println("Network loaded successfully!\n");
                }
                else
                {
                    System.out.println("Problem loading net.");
                }
                break;
            case "3":   // Calculates accuracy on training data
                if (hasWeights) {
                    System.out.println("\nGetting accuracy on TRAINING DATA...\n");
                    neuralNet.getAccuracy(TRAINING_FILE, TRAINING_IMAGES, "none");
                }
                else
                    invalidInput();
                break;
            case "4":   // Calculates accuracy on testing data
                if (hasWeights)
                {
                    System.out.println("\nGetting accuracy on TESTING DATA...\n");
                    neuralNet.getAccuracy(TESTING_FILE, TESTING_IMAGES, "none");
                }
                else
                    invalidInput();
                break;
            case "5":   // Saves the current network to a file
                if (hasWeights)
                {
                    System.out.println("\nAttempting to save network...\n");
                    if (!neuralNet.save(WEIGHTS_FILE)) {
                        System.out.println("Problem saving net.");
                    }
                    else
                    {
                        System.out.println("\nNetwork saved successfully!\n");
                    }
                }
                else
                    invalidInput();
                break;
            case "6":   // Graphically reviews all testing data
                if (hasWeights)
                {
                    System.out.println("\nPreparing graphical analysis of all TESTING DATA...\n");
                    neuralNet.getAccuracy(TESTING_FILE, TESTING_IMAGES, "all");
                }
                else
                    invalidInput();
                break;
            case "7":   // Graphically reviews only incorrectly classified testing data
                if (hasWeights)
                {
                    System.out.println("\nPreparing graphical analysis of incorrectly classified TESTING DATA...\n");
                    neuralNet.getAccuracy(TESTING_FILE, TESTING_IMAGES, "incorrect");
                }
                else
                    invalidInput();
                break;
            case "0":   // Exits the program
                System.out.print("Bye");
                System.exit(0);
            default:    // Notifies the user that their input is invalid
                invalidInput();
                break;
        }
    }

    // Simple function to notify the user that their input was invalid
    private void invalidInput()
    {
        System.out.println("\n\nInvalid input. Press enter a valid option.\n\n");
    }

    // I could not decide whether or not to use this function...
    private void clearScreen()
    {
        ;
    }

    // Gets the user's input from stdin
    public static String getInput()
    {
        Scanner scan = new Scanner(System.in);
        return scan.next();
    }

    // Main method of the front end
    // Continuously loops through the menu and processes the user's input
    public void run()
    {
        System.out.println("Welcome to George's MNIST Neural Net!");
        String input;
        while (true)
        {
            input = menu();
            processInput(input);
        }
    }
}

class Network
{
    // Sizes of the layers
    // Passed in from the front end
    private final int[] SIZES;

    private final Matrix[] biases;
    private final Matrix[] weights;
    private final int layers;

    // Constructor for the network object
    public Network(int[] sizes)
    {
        SIZES = sizes;

        layers = SIZES.length;
        biases = new Matrix[layers];
        weights = new Matrix[layers];

        // builds the weight and bias matrices
        for (int i = 1; i < layers; i++)
        {
            biases[i] = new Matrix(SIZES[i], 1);
            biases[i].randomize(SIZES[0] + SIZES[layers - 1]);
            weights[i] = new Matrix(SIZES[i], SIZES[i - 1]);
            weights[i].randomize(SIZES[0] + SIZES[layers - 1]);
        }
    }

    // generates an ASCII representation of the hand drawn number
    private String drawNumber(Matrix a)
    {
        // grey scale to ascii spectrum
        String spectrum = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";
        int lineLength = 28;

        // IntelliJ suggested I use StringBuilder instead of repeated string concatenations
        StringBuilder numPic = new StringBuilder();

        // Builds a string that represents the hand-written digit
        for (int i = 0; i < SIZES[0]; i++)
        {
            if (i % lineLength == 0)
            {
                numPic.append("\n");
            }

            // converts pixel value to spectrum index
            int val = (int)Math.ceil(((spectrum.length() - 1) * (a.array[i][0])));
            numPic.append(spectrum.charAt(spectrum.length() - 1 - val));
        }

        System.out.println(numPic);

        // gives user options
        // actually, anything but 0 will continue...
        System.out.println("[0] Return to main menu");
        System.out.println("[1] Continue");

        // gets user input
        return FrontEnd.getInput();
    }

    // Gets accuracy on either testing or training data
    public void getAccuracy(String fileName, int numImages, String mode)
    {
        // gets data to test
        int[][] testData = getData(fileName, numImages);

        // total of each digit
        int totals[] = new int[10];
        // correct number of each digit
        int correct[] = new int[10];

        // iterates through each image
        for (int i = 0; i < numImages; i++)
        {
            // builds input vector
            Matrix input = new Matrix(SIZES[0], 1);
            for (int p = 1; p < SIZES[0]; p++)
            {
                input.array[p][0] = testData[i][p + 1];
            }

            // gets output from feedForward
            Matrix a[] = feedForward(input);

            // updates totals and correct vectors
            totals = updateTotals(totals, testData[i][0]);
            correct = updateCorrect(correct, a[layers-1], testData[i][0]);

            // gets calculated output
            int output = classifyOutput(a[layers - 1]);

            // determines if the number should be drawn
            if (Objects.equals(mode, "all") || (Objects.equals(mode, "incorrect") && testData[i][0] != output))
            {
                System.out.println("Correct Classification:\t" + testData[i][0]);
                System.out.println("System Classification:\t" + output);
                System.out.println("Verdict:\t" + ((output == testData[i][0]) ? "Correct" : "Incorrect"));

                // draws the number
                String response = drawNumber(input);

                // return to main menu if user selects "0"
                if (response.equals("0"))
                {
                    return ;
                }
            }
        }

        printStats(correct, totals);
    }

    // Prints the statistics table
    private void printStats(int[] correct, int[] totals)
    {
        System.out.println("number\tcorrect\ttotal\tpercent");  // header

        for (int index = 0; index < 10; index++)    // iterates through each digit
        {
            System.out.printf("%s\t\t%s \t%s \t%.4f\n",
                    index, correct[index], totals[index], 100.0 * (double)correct[index] / totals[index]);
        }
        System.out.printf("Total\t%s \t%s \t%.4f\n\n",
                sum(correct), sum(totals), 100.0 * (double)sum(correct) / sum(totals));
    }

    // Fisher-Yates shuffle taken from PhiLho at Stack Overflow
    private int[] shuffle(int[] arr)
    {
        Random randy = new Random();
        for (int i = arr.length - 1; i > 0; i--)
        {
            int index = randy.nextInt(i + 1);
            int a = arr[index];
            arr[index] = arr[i];
            arr[i] = a;
        }
        return arr;
    }

    // Stochastic Gradient Descent function
    public void SGD(int epochs, int batch_size, double eta, String fileName, int numImages)
    {
        int[][] testData = getData(fileName, numImages);

        int[] orderedList = new int[numImages];
        for (int i = 0; i < numImages; i++)
        {
            orderedList[i] = i;
        }

        // Performs SCD for the given number of epochs
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // keeps track of the total count of each digit and the correct count of each digit
            int totals[] = new int[10];
            int correct[] = new int[10];

            // Shuffles the order of the data
            orderedList = shuffle(orderedList);
            // finds total number of batches
            int numBatches = numImages / batch_size;

            int shuffleIndex = 0;  // index for the shuffled images

            // perform SGD for each mini batch
            for (int batch = 0; batch < numBatches; batch++)
            {
                // gradient matrices
                Matrix weightGradients[] = new Matrix[layers];
                Matrix biasGradients[] = new Matrix[layers];

                // initialize gradient matrices to 0
                for (int m = 1; m < layers; m++)
                {
                    weightGradients[m] = new Matrix(weights[m].rows, weights[m].cols);
                    weightGradients[m] = weightGradients[m].fill(0);
                    biasGradients[m] = new Matrix(biases[m].rows, biases[m].cols);
                    biasGradients[m] = biasGradients[m].fill(0);
                }

                int i;

                // for each image in the mini batch
                for (int subIndex = 0; subIndex < batch_size; subIndex++, shuffleIndex++)
                {
                    i = orderedList[shuffleIndex];
                    // builds desired output
                    Matrix Y = buildY(testData[i][0]);

                    // builds input matrix
                    Matrix input = new Matrix(SIZES[0], 1);
                    for (int p = 1; p < SIZES[0]; p++)
                    {
                        input.array[p][0] = testData[i][p + 1];
                    }

                    // executes feed forward process
                    Matrix a[] = feedForward(input);

                    // updates totals
                    totals = updateTotals(totals, testData[i][0]);
                    correct = updateCorrect(correct, a[layers-1], testData[i][0]);

                    // gets error values for each layer
                    Matrix error[] = getError(a, Y);

                    // fills bias and weight gradient matrices using the error
                    for (int L = 1; L < layers; L++)
                    {
                        biasGradients[L] = biasGradients[L].add(error[L]);
                        for (int j = 0; j < weights[L].rows; j++)
                        {
                            for (int k = 0; k < weights[L].cols; k++)
                            {
                                weightGradients[L].array[j][k] += a[L-1].array[k][0] * error[L].array[j][0];
                            }
                        }

                    }

                }

                // adjusts the weights and biases using the gradients
                for (int L = 1; L < layers; L++)
                {
                    weights[L] = weights[L].subtract(weightGradients[L].scalarMultiply((eta / batch_size)));
                    biases[L] = biases[L].subtract(biasGradients[L].scalarMultiply(eta / batch_size));
                }

            }

            // I've found that this works better with larger hidden layers...
            // eta /= 2;

            // prints statistics for the current epoch
            System.out.println("Epoch: " + (epoch + 1));
            printStats(correct, totals);
        }

    }

    // generates the sum on an array of integers
    private int sum(int arr[])
    {
        int total = 0;
        // For each element in arr
        for (int element : arr)
        {
            total += element;
        }
        return total;
    }

    // Updates the appropriate value in the totals vector
    private int[] updateTotals(int totals[], int Y)
    {
        totals[Y]++;
        return totals;
    }

    // Updates the appropriate value in the correct vector
    private int[] updateCorrect(int correct[], Matrix a, int y)
    {
        int output;

        // determines the output of the system
        output = classifyOutput(a);

        // update the correct vector if the expected value and the generated value match
        if (output == y)
        {
            correct[output] += 1;
        }

        return correct;
    }

    // Returns the index of the highest value of the output vector
    private int classifyOutput(Matrix a)
    {
        double max = 0;
        int index = -1;

        // finds the largest value in the output vector
        for (int i = 0; i < 10; i++)
        {
            // update if a larger value is found
            if (a.array[i][0] > max)
            {
                max = a.array[i][0];
                index = i;
            }
        }

        return index;
    }

    // using the expected output digit, builds the expected output vector
    private Matrix buildY(int number)
    {
        Matrix y = new Matrix(10, 1);
        y = y.fill(0);
        y.array[number][0] = 1; // the value of the index of the correct output is set to 1
        return y;
    }

    // performs the feed forward algorithm
    private Matrix[] feedForward(Matrix input)
    {
        Matrix a[] = new Matrix[layers];

        a[0] = input;

        // normalizes the input data to a range [0,1]
        for (int i = 0; i < SIZES[0]; i++)
        {
            a[0].array[i][0] /= 255;
        }

        // computes the output of each layer of the network
        for (int i = 1; i < layers; i++)
        {
            a[i] = weights[i].multiply(a[i-1]).add(biases[i]).sigmoid();
        }

        // returns the output of each layer
        return a;
    }

    // computes the error at each layer
    private Matrix[] getError(Matrix[] a, Matrix Y)
    {
        // initializes error vectors to 0
        Matrix error[] = new Matrix[layers];
        for (int i = 0; i < layers; i++)
        {
            error[i] = new Matrix(a[i].rows, a[i].cols);
            error[i] = error[i].fill(0);
        }

        int L = layers - 1;

        // creates a vector of all 1's
        Matrix one = new Matrix(a[L].rows, a[L].cols);
        one = one.fill(1);

        // computes error of output layer
        error[L] = a[L].subtract(Y).hadamard(a[L]).hadamard(one.subtract(a[L]));

        // computes error of other layers
        for (L = layers - 2; L > 0; L--)
        {
            one = a[L].fill(1);
            error[L] = weights[L+1].transpose().multiply(error[L+1]).hadamard(a[L].hadamard(one.subtract(a[L])));
        }

        // returns the error at each layer
        return error;
    }

    // generates a data set from an input file
    private int[][] getData(String fileName, int numImages)
    {
        int dataSet[][] = new int[numImages][SIZES[0] + 1]; // +1 for classification
        BufferedReader inputReader;

        try
        {
            inputReader = new BufferedReader(new FileReader(fileName));
            String line[];
            // reads the appropriate number of images
            for(int i = 0; i < numImages; i++)
            {
                line = inputReader.readLine().split(",");

                // converts each string number to an int and adds it to the data set
                for(int j = 0; j < line.length; j++)
                {
                    dataSet[i][j] = Integer.parseInt((line[j]));
                }
            }
            inputReader.close();
        }
        catch (Exception e)
        {
            System.out.println(e);
        }

        // returns the generated data set
        return dataSet;
    }

    // saves the current weights and biases to a file
    public boolean save(String weightsFile)
    {
        BufferedWriter outputWriter;
        try
        {
            outputWriter = new BufferedWriter(new FileWriter(weightsFile));
        }
        catch (Exception e)
        {
            return false;
        }

        StringBuilder outputString = new StringBuilder();

        // for each layer in the network after the input layer
        for (int layer = 1; layer < weights.length; layer++)
        {
            double[][] layerWeights;
            double[][] layerBiases;
            layerWeights = weights[layer].array;
            layerBiases = biases[layer].array;

            // write the weight matrix to a csv
            for (int i = 0; i < weights[layer].rows; i++)
            {
                for (int j = 0; j < weights[layer].cols - 1; j++)
                {
                    outputString.append(Double.toString(layerWeights[i][j])).append(",");
                }
                outputString.append(Double.toString(layerWeights[i][weights[layer].cols - 1])).append("\n");
            }
            outputString.append("\n");

            // write the bias vector to a csv
            for (int i = 0; i < biases[layer].rows; i++)
            {
                outputString.append(Double.toString(layerBiases[i][0])).append("\n");
            }
            outputString.append("\n");
        }
        try{
            outputWriter.write(outputString.toString());
            outputWriter.close();
        }
        catch (Exception e)
        {
            return false;
        }

        // return true if the write was successful
        return true;
    }

    // reads a set of weights and biases from a file
    public boolean load(String weightsFile)
    {
        String[] line;
        BufferedReader inputReader;

        try
        {
            inputReader = new BufferedReader(new FileReader(weightsFile));

            // for each layer in the network after the first
            for (int layer = 1; layer < weights.length; layer++)
            {
                // read the weight matrix
                for (int i = 0; i < weights[layer].rows; i++)
                {
                    line = inputReader.readLine().split(",");

                    for (int j = 0; j < weights[layer].cols; j++)
                    {
                        weights[layer].array[i][j] = Double.parseDouble(line[j]);
                    }
                }
                inputReader.readLine();

                // read the bias vector
                for (int i = 0; i < biases[layer].rows; i++)
                {
                    line = inputReader.readLine().split(",");
                    biases[layer].array[i][0] = Double.parseDouble(line[0]);
                }
                inputReader.readLine();
            }
            inputReader.close();
        }
        catch (Exception e)
        {
            return false;
        }

        // returns true if the load was successful
        return true;
    }
}


// a class used to perform matrix arithmetic
class Matrix
{
    // each matrix has a number of rows, a number of columns, and an underlying array
    final int rows;
    final int cols;
    final double[][] array;

    // each matrix is initialized with a number of rows and columns
    public Matrix(int r, int c)
    {
        rows = r;
        cols = c;
        array = new double[r][c];
    }

    // performs the sigmoid function on a scalar
    private double sigmoidFunc(double z)
    {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    // performs the sigmoid function on every element of the matrix
    public Matrix sigmoid()
    {
        Matrix sigmoidMatrix = new Matrix(this.rows, this.cols);

        for (int i = 0; i < this.rows; i++)
        {
            for (int j = 0; j < this.cols; j++)
            {
                sigmoidMatrix.array[i][j] = sigmoidFunc(this.array[i][j]);
            }
        }

        return sigmoidMatrix;
    }

    // transposes the matrix
    // Ai,j = Aj,i
    public Matrix transpose()
    {
        Matrix transposeMatrix = new Matrix(this.cols, this.rows);
        for (int i = 0; i < transposeMatrix.rows; i++)
        {
            for (int j = 0; j < transposeMatrix.cols; j++)
            {
                transposeMatrix.array[i][j] = this.array[j][i];
            }
        }

        return transposeMatrix;
    }

    // performs matrix multiplication
    public Matrix multiply(Matrix other)
    {
        // checks for a dim mismatch
        if (this.cols != other.rows)
        {
            System.out.println("Error: Matrix dim mismatch");
            System.exit(1);
        }

        // straight forward matrix multiplication
        Matrix product = new Matrix(this.rows, other.cols);
        for (int i = 0; i < product.rows; i++)
        {
            for (int j = 0; j < product.cols; j++)
            {
                double dotProd = 0;
                for (int k = 0; k < this.cols; k++)
                {
                    dotProd = dotProd + (this.array[i][k] * other.array[k][j]);
                }

                product.array[i][j] = dotProd;
            }
        }
        return product;
    }

    // multiplies every element of the matrix by a scalar
    public Matrix scalarMultiply(double s)
    {
        Matrix scaledMatrix = new Matrix(this.rows, this.cols);

        for (int i = 0; i < this.rows; i++)
        {
            for (int j = 0; j < this.cols; j++)
            {
                scaledMatrix.array[i][j] = this.array[i][j] * s;
            }
        }

        return scaledMatrix;
    }

    // performs element by element multiplication between 2 matrices
    public Matrix hadamard(Matrix other)
    {
        // checks for a dim mismatch
        if (this.rows != other.rows || this.cols != other.cols) {
            System.out.println("Error: Hadamard dim mismatch");
            System.exit(1);
        }

        Matrix hadamardMatrix = new Matrix(this.rows, this.cols);

        for (int i = 0; i < this.rows; i++)
        {
            for (int j = 0; j < this.cols; j++)
            {
                hadamardMatrix.array[i][j] = this.array[i][j] * other.array[i][j];
            }
        }

        return hadamardMatrix;
    }

    // straight forward matrix addition
    public Matrix add(Matrix other)
    {
        if (this.rows != other.rows || this.cols != other.cols)
        {
            System.out.println("Error: Matrix dim mismatch (+)");
            System.exit(1);
        }

        Matrix sum = new Matrix(this.rows, this.cols);
        for (int i = 0; i < sum.rows; i++)
        {
            for (int j = 0; j < sum.cols; j++)
            {
                sum.array[i][j] = this.array[i][j] + other.array[i][j];
            }
        }
        return sum;
    }

    // A + (-B)
    public Matrix subtract(Matrix other)
    {
        return this.add(other.scalarMultiply(-1));
    }

    // fills a matrix with random numbers
    // uniform distribution with range [-r, r]
    // r = sqrt(6 / (inputNodes + outputNodes))
    public void randomize(double fan_in_out)
    {
        // My high school java textbook named every random object "randy"
        Random randy = new Random();

        double r =  Math.sqrt(6 / fan_in_out);

        // fill every element with a random double
        for (int i = 0; i < this.rows; i++)
        {
            for (int j = 0; j < this.cols; j++)
            {
                this.array[i][j] = randy.nextDouble() * 2 * r - r;
            }
        }
    }

    // returns a matrix of the same dimension filled with the given number
    // useful for generating a matrix where every element is the same
    public Matrix fill(double c)
    {
        Matrix filled = new Matrix(this.rows, this.cols);
        for (int i = 0; i < filled.rows; i++)
        {
            for (int j = 0; j < filled.cols; j++)
            {
                filled.array[i][j] = c;
            }
        }
        return filled;
    }
}

