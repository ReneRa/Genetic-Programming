package core;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;
import java.util.StringTokenizer;

import utils.Utils;

public class Main {

	public static final boolean SHUFFLE_AND_SPLIT = true;
	// public static final boolean SHUFFLE_AND_SPLIT = false;

	public static final String DATA_FILENAME = "dataset";

	public static final int NUMBER_OF_RUNS = 30;
	public static final int NUMBER_OF_GENERATIONS = 2000;
	
	public static Population bestIndividualAtGenerations = new Population();
	public static int[] stopAtGen = new int[NUMBER_OF_RUNS];
	public static double[][] trainingErrors = new double[NUMBER_OF_RUNS][NUMBER_OF_GENERATIONS+1];
	public static double[][] unseenErrors = new double[NUMBER_OF_RUNS][NUMBER_OF_GENERATIONS+1];
	public static int[][] sizes = new int[NUMBER_OF_RUNS][NUMBER_OF_GENERATIONS+1];
	public static int[][] depths = new int[NUMBER_OF_RUNS][NUMBER_OF_GENERATIONS+1];
	public static int CURRENT_RUN;

	public static void main(String[] args) {

		// load training and unseen data
		Data data = loadData(DATA_FILENAME);

		// run GP for a given number of runs
		double[][] resultsPerRun = new double[4][NUMBER_OF_RUNS];
		for (int i = 0; i < NUMBER_OF_RUNS; i++) {
			System.out.printf("\n\t\t##### Run %d #####\n", i + 1);
			CURRENT_RUN = i;
			
			GpRun gp = new GpRun(data);

//			GsgpRun gp = new GsgpRun(data);
//			gp.setBuildIndividuals(true);
//			gp.setBoundedMutation(true);

			Individual bestFound = gp.evolve(NUMBER_OF_GENERATIONS);

			resultsPerRun[0][i] = bestFound.getTrainingError();
			resultsPerRun[1][i] = bestFound.getUnseenError();
			resultsPerRun[2][i] = bestFound.getSize();
			resultsPerRun[3][i] = bestFound.getDepth();
			System.out.print("\nBest =>");
			bestFound.print();
			System.out.println();
			// write individual to object file
			bestFound.writeToObjectFile();
		}

		// present average results
		System.out.printf("\n\t\t##### Results #####\n\n");
		System.out.printf("Average training error:\t\t%.2f\n", Utils.getAverage(resultsPerRun[0]));
		System.out.printf("Average unseen error:\t\t%.2f\n", Utils.getAverage(resultsPerRun[1]));
		System.out.printf("Average size:\t\t\t%.2f\n", Utils.getAverage(resultsPerRun[2]));
		System.out.printf("Average depth:\t\t\t%.2f\n", Utils.getAverage(resultsPerRun[3]));
	
		try {
			saveDataToFile();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	// Saves average data from runs over all generations to a file. 
	// This allows analysis of the results through plotting the results
	private static void saveDataToFile() throws IOException {
		double[] avgTrainingErrorAr = new double[NUMBER_OF_GENERATIONS];
		double[] avgUnseenErrorAr = new double[NUMBER_OF_GENERATIONS];
		int[] avgSizeAr = new int[NUMBER_OF_GENERATIONS];
		int[] avgDepthAr = new int[NUMBER_OF_GENERATIONS];
		
		for (int generation = 0; generation < NUMBER_OF_GENERATIONS; generation++){
			double avgTrainingError = 0.0;
			double avgUnseenError = 0.0;
			int avgSize = 0;
			int avgDepth = 0;
			
			for (int run = 0; run < NUMBER_OF_RUNS; run++){
				avgTrainingError += trainingErrors[run][generation];
				avgUnseenError += unseenErrors[run][generation];
				avgSize += sizes[run][generation];
				avgDepth += depths[run][generation];
			}
			
			avgTrainingErrorAr[generation] = avgTrainingError / NUMBER_OF_RUNS;
			avgUnseenErrorAr[generation] = avgUnseenError / NUMBER_OF_RUNS;
			avgSizeAr[generation] = avgSize / NUMBER_OF_RUNS;
			avgDepthAr[generation] = avgDepth / NUMBER_OF_RUNS;
		}
		String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(Calendar.getInstance().getTime());
		
		File file = new File(
				"data/" +  timeStamp +  ".txt");

		// if file doesn't exists, then create it
		if (!file.exists()) {
			file.createNewFile();
		}

		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);

		bw.write(
				"Generation \t Training Error \t Unseen Error \t Size \t Depth \n");

		for (int i = 0; i < Main.NUMBER_OF_GENERATIONS; i++) {
			bw.write(i + "\t" + String.valueOf(avgTrainingErrorAr[i]).replace('.', ',') + "\t" 
					+ String.valueOf(avgUnseenErrorAr[i]).replace('.', ',') + "\t"
					+ String.valueOf(avgSizeAr[i]) + "\t"
					+ String.valueOf(avgDepthAr[i]));
			if(i != Main.NUMBER_OF_GENERATIONS-1){bw.write("\n");}
		}
		bw.close();
		
		
	}

	public static Data loadData(String dataFilename) {
		double[][] trainingData, unseenData;

		if (SHUFFLE_AND_SPLIT) {
			double[][] allData = readData(dataFilename + ".txt");
			List<Integer> instances = Utils.shuffleInstances(allData.length);
			int trainingInstances = (int) Math.floor(0.7 * allData.length);
			int unseenInstances = (int) Math.ceil(0.3 * allData.length);

			trainingData = new double[trainingInstances][];
			unseenData = new double[unseenInstances][];

			for (int i = 0; i < trainingInstances; i++) {
				trainingData[i] = allData[instances.get(i)];
			}

			for (int i = 0; i < unseenInstances; i++) {
				unseenData[i] = allData[instances.get(trainingInstances + i)];
			}
		} else {
			trainingData = readData(dataFilename + "_training.txt");
			unseenData = readData(dataFilename + "_unseen.txt");
		}
		return new Data(trainingData, unseenData);
	}

	public static double[][] readData(String filename) {
		double[][] data = null;
		List<String> allLines = new ArrayList<String>();
		try {
			BufferedReader inputBuffer = new BufferedReader(new FileReader(filename));
			String line = inputBuffer.readLine();
			while (line != null) {
				allLines.add(line);
				line = inputBuffer.readLine();
			}
			inputBuffer.close();
		} catch (Exception e) {
			System.out.println(e);
		}

		StringTokenizer tokens = new StringTokenizer(allLines.get(0).trim());
		int numberOfColumns = tokens.countTokens();
		data = new double[allLines.size()][numberOfColumns];
		for (int i = 0; i < data.length; i++) {
			tokens = new StringTokenizer(allLines.get(i).trim());
			for (int k = 0; k < numberOfColumns; k++) {
				data[i][k] = Double.parseDouble(tokens.nextToken().trim());
			}
		}
		return data;
	}
}
