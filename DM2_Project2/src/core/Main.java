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

	public static final int NUMBER_OF_RUNS = 10;
	public static final int NUMBER_OF_GENERATIONS = 500;
	
	public static Population bestIndividualAtGenerations = new Population();
	public static int[] stopAtGen = new int[NUMBER_OF_RUNS];
	public static double[][] trainingErrors = new double[NUMBER_OF_RUNS][NUMBER_OF_GENERATIONS+1];
	public static double[][] unseenErrors = new double[NUMBER_OF_RUNS][NUMBER_OF_GENERATIONS+1];
	public static int[][] sizes = new int[NUMBER_OF_RUNS][NUMBER_OF_GENERATIONS+1];
	public static int[][] depths = new int[NUMBER_OF_RUNS][NUMBER_OF_GENERATIONS+1];
	public static int CURRENT_RUN;
	
	protected static boolean seedInitialization = false;
	protected static boolean seedMutatedInitialization = false;
	protected static int numberOfSeedIterations = 100;
	protected static double percentageOfSeedIndividuals = 5.0;
	protected static String executableAlgorithm = "GP"; // Either GP or GSGP
	
	protected static boolean interleavedSampling = true;
	//uniformCrossover,onePointCrossover,standardCrossover;	public static final String selectedCrossoverMethod = "standardCrossover"; 
	public static void main(String[] args) {

		// load training and unseen data
		Data data = loadData(DATA_FILENAME);
		
		// run GP for a given number of runs
		double[][] resultsPerRun = new double[4][NUMBER_OF_RUNS];
		for (int i = 0; i < NUMBER_OF_RUNS; i++) {
			System.out.printf("\n\t\t##### Run %d #####\n", i + 1);
			CURRENT_RUN = i;
			
			Individual bestFound = new Individual();
			
			
			if(seedInitialization == false){
				if(executableAlgorithm.equals("GP")){
					GpRun gp = new GpRun(data, interleavedSampling);
					bestFound = gp.evolve(NUMBER_OF_GENERATIONS);
				}
				else if(executableAlgorithm.equals("GSGP")){
					GsgpRun gsgp = new GsgpRun(data, interleavedSampling);
					bestFound = gsgp.evolve(NUMBER_OF_GENERATIONS);
				}
			}
			else{
				if(executableAlgorithm.equals("GP")){
					GpRun seedGP = new GpRun(data, interleavedSampling);
					Population seedPopulation = new Population();
					seedGP.evolve(numberOfSeedIterations);
					seedPopulation = seedGP.getPopulation();
					GpRun gp = new GpRun(data, interleavedSampling);
					if(seedMutatedInitialization == true){gp.population = getMutatedSeededPopulation(gp, data, seedPopulation);}
					else{gp.population = getSeededPopulation(seedPopulation, gp.getPopulation());}
					bestFound = gp.evolve(NUMBER_OF_GENERATIONS);
				}
				else if(executableAlgorithm.equals("GSGP")){
					GpRun seedGP = new GpRun(data, interleavedSampling);
					Population seedPopulation = new Population();
					seedGP.evolve(numberOfSeedIterations);
					seedPopulation = seedGP.getPopulation();
					GsgpRun gsgp = new GsgpRun(data, interleavedSampling);
					if(seedMutatedInitialization == true){gsgp.population = getMutatedSeededPopulation(gsgp, data, seedPopulation);}
					else{gsgp.population = getSeededPopulation(seedPopulation, gsgp.getPopulation());}
					bestFound = gsgp.evolve(NUMBER_OF_GENERATIONS);
				}
			}
			// gp.setBuildIndividuals(true);
			// gp.setBoundedMutation(true);


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
	
	// Seeds the initial population with the best individual from the initial population and it's mutations
	private static Population getMutatedSeededPopulation(GpRun gp, Data data, Population seedPopulation){
		Population population = new Population();
		Individual bestIndividual = seedPopulation.getBest();
		
		population.addIndividual(bestIndividual);
		
		for(int i = 1; i < gp.getPopulationSize(); i++){
			population.addIndividual(gp.applyStandardMutation(bestIndividual));
			population.getIndividual(i).evaluate(data);
		}
		
		return population;
	}
	
	private static Population getMutatedSeededPopulation(GsgpRun gsgp, Data data, Population seedPopulation){
		Population population = new Population();
		Individual bestIndividual = seedPopulation.getBest();
		
		population.addIndividual(bestIndividual);
		
		for(int i = 1; i < gsgp.getPopulationSize(); i++){
			population.addIndividual(gsgp.applyStandardMutation(bestIndividual));
			population.getIndividual(i).evaluate(data);
		}
		
		return population;
	}
	
	// Replaces the worst individuals from the gsgp population with the best from the seed individuals.
	private static Population getSeededPopulation(Population seedPopulation, Population gsgpPopulation) {
		int amountOfSeededIndividuals = (int) Math.round(gsgpPopulation.getSize() * (percentageOfSeedIndividuals/100));
		
		// Get the best individuals from the GP run that will be used as seed
		ArrayList<Individual> seedIndividuals = seedPopulation.getBestIndividuals(amountOfSeededIndividuals);
		// Get the best individuals from the GSGP run that will make up the rest of the population
		ArrayList<Individual> bestGSGPIndividuals = gsgpPopulation.getBestIndividuals(gsgpPopulation.getSize() - amountOfSeededIndividuals);
		
		Population population = new Population();
		
		// Add the seed individuals and the gsgp individuals to the population respectively
		for(int i = 0; i < seedIndividuals.size(); i ++){
			population.addIndividual(seedIndividuals.get(i));
		}
		for(int i = 0; i < bestGSGPIndividuals.size(); i ++){
			population.addIndividual(bestGSGPIndividuals.get(i));
		}
		return population;
		
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
