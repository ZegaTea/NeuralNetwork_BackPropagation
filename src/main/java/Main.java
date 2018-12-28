import neuron.NeuralNetwork;
import process.Process;
import util.CommonUtils;
import process.*;
import util.ConfigurationManager;


public class Main {

    private static double[][] trainData = null;
    private static double[][] testData = null;

    public static void main(String[] args) throws Exception {
        String fileName = ConfigurationManager.getInstance().getString("file.data");
        Process process = new Process();
        double[][] allData = process.getInput(fileName);
        makeTrainTest(allData);

        final int numInput = 4;
        final int numHidden = 7;
        final int numOutput = 3;
        NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);

        nn.initializeWeights();

        int maxEpochs = 500;
        double learnRate = 0.05;
        double momentum = 0.01;

        nn.train(trainData, maxEpochs, learnRate, momentum); // back-propagation

        double[] weights = nn.getWeights();

        double trainAcc = nn.accuracy(trainData);
        System.out.println("\nAccuracy on training data = " + trainAcc);

        double testAcc = nn.accuracy(testData);
        System.out.println("\nAccuracy on test data = " + testAcc);

    }

    private static void makeTrainTest(double[][] allData) {
        // split allData into 80% trainData and 20% testData
        int totRows = allData.length;
        int numCols = allData[0].length;

        int trainRows = (int) (totRows * 0.80); // hard-coded 80-20 split
        int testRows = totRows - trainRows;

        trainData = new double[trainRows][];
        testData = new double[testRows][];

        int[] sequence = new int[totRows]; // create a random sequence of indexes
        for (int i = 0; i < sequence.length; ++i) {
            sequence[i] = i;
        }

        for (int i = 0; i < sequence.length; ++i) {
            int r = CommonUtils.randomInt(i, sequence.length);
            int tmp = sequence[r];
            sequence[r] = sequence[i];
            sequence[i] = tmp;
        }

        int si = 0; // index into sequence[]
        int j = 0; // index into trainData or testData

        for (; si < trainRows; ++si) // first rows to train data
        {
            trainData[j] = new double[numCols];
            int idx = sequence[si];
            System.arraycopy(allData[idx], 0, trainData[j], 0, numCols);
            ++j;
        }

        j = 0; // reset to start of test data
        for (; si < totRows; ++si) // remainder to test data
        {
            testData[j] = new double[numCols];
            int idx = sequence[si];
            System.arraycopy(allData[idx], 0, testData[j], 0, numCols);
            ++j;
        }
    } // makeTrainTest
}