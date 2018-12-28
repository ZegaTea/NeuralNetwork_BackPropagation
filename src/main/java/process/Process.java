package process;

import util.FileUtils;

import java.util.List;

public class Process {
    public double[][] getInput(String fileName) {
        FileUtils fileUtils = new FileUtils();
        List<String> data = fileUtils.readFile(fileName);
        double[][] allData = new double[data.size()][];

        for (int i = 0; i < allData.length; i++) {
            String line[] = data.get(i).split(",");
            allData[i] = new double[line.length];
            for (int j = 0; j < allData[i].length; j++) {
                allData[i][j] = Double.parseDouble(line[j]);
            }
        }


        return allData;
    }
}
