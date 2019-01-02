package neuron;

import util.CommonUtils;

import java.util.Random;

public class NeuralNetwork {

    private static Random rnd;

    private int numInput;
    private int numHidden;
    private int numOutput;

    private double[] inputs;

    private double[][] ihWeights; // input-hidden
    private double[] hBiases;
    private double[] hOutputs;

    private double[][] hoWeights; // hidden-output
    private double[] oBiases;

    private double[] outputs;

    // back-prop specific arrays (these could be local to updateWeights)
    private double[] oGrads; // output gradients for back-propagation
    private double[] hGrads; // hidden gradients for back-propagation

    // back-prop momentum specific arrays (necessary as class members)
    private double[][] ihPrevWeightsDelta;  // for momentum with back-propagation
    private double[] hPrevBiasesDelta;
    private double[][] hoPrevWeightsDelta;
    private double[] oPrevBiasesDelta;

    public NeuralNetwork(int numInput, int numHidden, int numOutput) {
        rnd = new Random(0); // for InitializeWeights() and Shuffle()

        this.numInput = numInput;
        this.numHidden = numHidden;
        this.numOutput = numOutput;

        this.inputs = new double[numInput];

        this.ihWeights = makeMatrix(numInput, numHidden);
        this.hBiases = new double[numHidden];
        this.hOutputs = new double[numHidden];

        this.hoWeights = makeMatrix(numHidden, numOutput);
        this.oBiases = new double[numOutput];

        this.outputs = new double[numOutput];

        // back-prop related arrays below
        this.hGrads = new double[numHidden];
        this.oGrads = new double[numOutput];

        this.ihPrevWeightsDelta = makeMatrix(numInput, numHidden);
        this.hPrevBiasesDelta = new double[numHidden];
        this.hoPrevWeightsDelta = makeMatrix(numHidden, numOutput);
        this.oPrevBiasesDelta = new double[numOutput];
    } // ctor

    private static double[][] makeMatrix(int rows, int cols) // helper for ctor
    {
        double[][] result = new double[rows][];
        for (int r = 0; r < result.length; ++r) {
            result[r] = new double[cols];
        }
        return result;
    }


    public final void setWeights(double[] weights) {
        //  copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
        int numWeights = ((numInput * numHidden)
                + ((numHidden * numOutput)
                + (numHidden + numOutput)));
        if ((weights.length != numWeights)) {
            return;
//            throw new Exception("Bad weights array length: ");
        }

        int k = 0;
        //  points into weights param
        for (int i = 0; (i < numInput); i++) {
            for (int j = 0; (j < numHidden); j++) {
                ihWeights[i][j] = weights[k++];
            }

        }

        for (int i = 0; (i < numHidden); i++) {
            hBiases[i] = weights[k++];
        }

        for (int i = 0; (i < numHidden); i++) {
            for (int j = 0; (j < numOutput); j++) {
                hoWeights[i][j] = weights[k++];
            }

        }

        for (int i = 0; (i < numOutput); i++) {
            oBiases[i] = weights[k++];
        }

    }

    public final void initializeWeights() {
        //  initialize weights and biases to small random values
        int numWeights = ((numInput * numHidden)
                + ((numHidden * numOutput)
                + (numHidden + numOutput)));
        double[] initialWeights = new double[numWeights];
        for (int i = 0; (i < initialWeights.length); i++) {
            initialWeights[i] = (((0.001 - 0.0001)
                    * rnd.nextDouble())
                    + 0.0001);
        }

        this.setWeights(initialWeights);
    }

    public final double[] getWeights() {
        //  returns the current set of wweights, presumably after training
        int numWeights = ((numInput * numHidden)
                + ((numHidden * numOutput)
                + (numHidden + numOutput)));
        double[] result = new double[numWeights];
        int k = 0;
        for (int i = 0; (i < ihWeights.length); i++) {
            for (int j = 0; (j < ihWeights[0].length); j++) {
                result[k++] = ihWeights[i][j];
            }

        }

        for (int i = 0; (i < hBiases.length); i++) {
            result[k++] = hBiases[i];
        }

        for (int i = 0; (i < hoWeights.length); i++) {
            for (int j = 0; (j < hoWeights[0].length); j++) {
                result[k++] = hoWeights[i][j];
            }

        }

        for (int i = 0; (i < oBiases.length); i++) {
            result[k++] = oBiases[i];
        }

        return result;
    }

    //  ----------------------------------------------------------------------------------------
    private final double[] computeOutputs(double[] xValues) throws Exception {
        if ((xValues.length != numInput)) {
            throw new Exception("Bad xValues array length");
        }

        double[] hSums = new double[numHidden];
        //  hidden nodes sums scratch array
        double[] oSums = new double[numOutput];
        //  output nodes sums
        for (int i = 0; (i < xValues.length); i++) {
            this.inputs[i] = xValues[i];
        }

        for (int j = 0; (j < numHidden); j++) {
            for (int i = 0; (i < numInput); i++) {
                hSums[j] = (hSums[j]
                        + (this.inputs[i] * this.ihWeights[i][j]));
            }

        }

        //  note +=
        for (int i = 0; (i < numHidden); i++) {
            hSums[i] = (hSums[i] + this.hBiases[i]);
        }

        for (int i = 0; (i < numHidden); i++) {
            this.hOutputs[i] = hyperTanFunction(hSums[i]);
        }

        //  hard-coded
        for (int j = 0; (j < numOutput); j++) {
            for (int i = 0; (i < numHidden); i++) {
                oSums[j] = (oSums[j]
                        + (hOutputs[i] * hoWeights[i][j]));
            }

        }

        for (int i = 0; (i < numOutput); i++) {
            oSums[i] = (oSums[i] + oBiases[i]);
        }

        double[] softOut = softmax(oSums);
        //  softmax activation does all outputs at once for efficiency
        System.arraycopy(softOut, 0, outputs, 0, softOut.length);

        double[] retResult = new double[numOutput];
        //  could define a GetOutputs method instead
        System.arraycopy(softOut, 0, retResult, 0, retResult.length);

        return retResult;
    }

    //  computeOutputs
    private static double hyperTanFunction(double x) {
        if ((x < -20)) {
            return -1;
        }

        //  approximation is correct to 30 decimals
        if ((x > 20)) {
            return 1;
        } else {
            return Math.tanh(x);
        }

    }

    private static double[] softmax(double[] oSums) {
        //  determine max output sum
        double max = oSums[0];
        for (int i = 0; (i < oSums.length); i++) {
            if ((oSums[i] > max)) {
                max = oSums[i];
            }

        }

        //  determine scaling factor -- sum of exp(each val - max)
        double scale = 0;
        for (int i = 0; (i < oSums.length); i++) {
            scale = (scale + Math.exp((oSums[i] - max)));
        }

        double[] result = new double[oSums.length];
        for (int i = 0; (i < oSums.length); i++) {
            result[i] = (Math.exp((oSums[i] - max)) / scale);
        }

        return result;
        //  now scaled so that xi sum to 1.0
    }

    //  ----------------------------------------------------------------------------------------
    private final void updateWeights(double[] tValues, double learnRate, double momentum) {
        //  update the weights and biases using back-propagation, with target values, eta (learning rate), alpha (momentum)
        //  assumes that setWeights and computeOutputs have been called and so all the internal arrays and matrices have values (other than 0.0)
        if ((tValues.length != numOutput)) {
            return;
//            throw new Exception("target values not same Length as output in updateWeights");
        }

        for (int i = 0; (i < oGrads.length); i++) {
            double derivative = ((1 - outputs[i])
                    * outputs[i]);
            //  derivative of softmax = (1 - y) * y (same as log-sigmoid)
            oGrads[i] = (derivative
                    * (tValues[i] - outputs[i]));
            //  'mean squared error version' includes (1-y)(y) derivative
            // oGrads[i] = (tValues[i] - outputs[i]); // cross-entropy version drops (1-y)(y) term! See http://www.cs.mcgill.ca/~dprecup/courses/ML/Lectures/ml-lecture05.pdf page 25.
        }

        //  2. compute hidden gradients
        for (int i = 0; (i < hGrads.length); i++) {
            double derivative = ((1 - hOutputs[i]) * (1 + hOutputs[i]));
            //  derivative of tanh = (1 - y) * (1 + y)
            double sum = 0;
            for (int j = 0; (j < numOutput); j++) {
                double x = (oGrads[j] * hoWeights[i][j]);
                sum = (sum + x);
            }

            hGrads[i] = (derivative * sum);
        }

        //  3a. update hidden weights (gradients must be computed right-to-left but weights can be updated in any order)
        for (int i = 0; (i < ihWeights.length); i++) {
            for (int j = 0; (j < ihWeights[0].length); j++) {
                double delta = (learnRate
                        * (hGrads[j] * inputs[i]));
                //  compute the new delta
                ihWeights[i][j] = (ihWeights[i][j] + delta);
                //  update. note we use '+' instead of '-'. this can be very tricky.
                ihWeights[i][j] = (ihWeights[i][j]
                        + (momentum * ihPrevWeightsDelta[i][j]));
                //  add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
                ihPrevWeightsDelta[i][j] = delta;
                //  don't forget to save the delta for momentum
            }

        }

        //  3b. update hidden biases
        for (int i = 0; (i < hBiases.length); i++) {
            double delta = (learnRate
                    * (hGrads[i] * 1));
            //  the 1.0 is the constant input for any bias; could leave out
            hBiases[i] = (hBiases[i] + delta);
            hBiases[i] = (hBiases[i]
                    + (momentum * hPrevBiasesDelta[i]));
            //  momentum
            hPrevBiasesDelta[i] = delta;
            //  don't forget to save the delta
        }

        //  4. update hidden-output weights
        for (int i = 0; (i < hoWeights.length); i++) {
            for (int j = 0; (j < hoWeights[0].length); j++) {
                double delta = (learnRate
                        * (oGrads[j] * hOutputs[i]));
                //  see above: hOutputs are inputs to the nn outputs
                hoWeights[i][j] = (hoWeights[i][j] + delta);
                hoWeights[i][j] = (hoWeights[i][j]
                        + (momentum * hoPrevWeightsDelta[i][j]));
                //  momentum
                hoPrevWeightsDelta[i][j] = delta;
                //  save
            }

        }

        //  4b. update output biases
        for (int i = 0; (i < oBiases.length); i++) {
            double delta = (learnRate
                    * (oGrads[i] * 1));
            oBiases[i] = (oBiases[i] + delta);
            oBiases[i] = (oBiases[i]
                    + (momentum * oPrevBiasesDelta[i]));
            //  momentum
            oPrevBiasesDelta[i] = delta;
            //  save
        }

    }

    //  updateWeights
    //  ----------------------------------------------------------------------------------------
    public final void train(double[][] trainData, int maxEprochs, double learnRate, double momentum) {
        //  train a back-prop style NN classifier using learning rate and momentum
        int epoch = 0;
        double[] xValues = new double[numInput];
        //  inputs
        double[] tValues = new double[numOutput];
        //  target values
        int[] sequence = new int[trainData.length];
        for (int i = 0; (i < sequence.length); i++) {
            sequence[i] = i;
        }

        while ((epoch < maxEprochs)) {
            double mse = 0;
            try {
                mse = meanSquaredError(trainData);
            } catch (Exception e) {
                e.printStackTrace();
            }
            if ((mse < 0.001)) {
                break;
            }

            //  consider passing value in as parameter
            shuffle(sequence);
            //  visit each training data in random order
            for (int i = 0; (i < trainData.length); i++) {
                int idx = sequence[i];
                System.arraycopy(trainData[idx], 0, xValues, 0, numInput);

                //  more flexible might be a 'GetInputsAndTargets()'
                System.arraycopy(trainData[idx], numInput, tValues, 0, numOutput);
                try {

                    computeOutputs(xValues);
                } catch (Exception e) {

                }
                //  copy xValues in, compute outputs (and store them internally)
                updateWeights(tValues, learnRate, momentum);
                //  use curr outputs and targets and back-prop to find better weights
            }

            //  each training tuple
            epoch++;
        }

    }

    //  train
    private static void shuffle(int[] sequence) {
        for (int i = 0; (i < sequence.length); i++) {
            int r = CommonUtils.randomInt(i, sequence.length);
            int tmp = sequence[r];
            sequence[r] = sequence[i];
            sequence[i] = tmp;
        }

    }

    private final double meanSquaredError(double[][] trainData) throws Exception {
        //  average squared error per training tuple
        double sumSquaredError = 0;
        double[] xValues = new double[numInput];
        //  first numInput values in trainData
        double[] tValues = new double[numOutput];
        //  last numOutput values
        for (int i = 0; (i < trainData.length); i++) {
            System.arraycopy(trainData[i], 0, xValues, 0, numInput);
            //  get xValues. more flexible would be a 'GetInputsAndTargets()'
            System.arraycopy(trainData[i], numInput, tValues, 0, numOutput);

            //  get target values
            double[] yValues = this.computeOutputs(xValues);
            //  compute output using current weights
            for (int j = 0; (j < numOutput); j++) {
                double err = (tValues[j] - yValues[j]);
                sumSquaredError = (sumSquaredError
                        + (err * err));
            }

        }

        return (sumSquaredError / trainData.length);
    }

    //  ----------------------------------------------------------------------------------------
    public final double accuracy(double[][] testData) throws Exception {
        //  percentage correct using winner-takes all
        int numCorrect = 0;
        int numWrong = 0;
        double[] xValues = new double[numInput];
        //  inputs
        double[] tValues = new double[numOutput];
        //  targets
        double[] yValues;
        //  computed Y
        for (int i = 0; (i < testData.length); i++) {
            System.arraycopy(testData[i], 0, xValues, 0, numInput);
            //  parse test data into x-values and t-values
            System.arraycopy(testData[i], numInput, tValues, 0, numOutput);

            yValues = this.computeOutputs(xValues);
            int maxIndex = maxIndex(yValues);
            //  which cell in yValues has largest value?
            if ((tValues[maxIndex] == 1)) {
                numCorrect++;
            } else {
                numWrong++;
            }

        }

        return (double) ((double)(numCorrect * 1)  / (double) (numCorrect + numWrong));
        //  ugly 2 - check for divide by zero
    }

    private static int maxIndex(double[] vector) {
        //  index of largest value
        int bigIndex = 0;
        double biggestVal = vector[0];
        for (int i = 0; (i < vector.length); i++) {
            if ((vector[i] > biggestVal)) {
                biggestVal = vector[i];
                bigIndex = i;
            }

        }

        return bigIndex;
    }

}
