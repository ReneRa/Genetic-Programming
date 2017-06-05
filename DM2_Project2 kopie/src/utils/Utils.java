package utils;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Utils {

	public static double calculateRMSE(double[][] data, double[] outputs) {
		double errorSum = 0.0;
		for (int i = 0; i < data.length; i++) {
			double target = data[i][data[0].length - 1];
			errorSum += Math.pow(outputs[i] - target, 2.0);
		}
		return Math.sqrt(errorSum / data.length);
	}

	public static Object readObjectFromFile(String filename) {
		Object object = null;
		try {
			InputStream inputStream = new BufferedInputStream(new FileInputStream(filename));
			ObjectInput objectInput = new ObjectInputStream(inputStream);
			try {
				object = objectInput.readObject();
			} finally {
				objectInput.close();
			}
		} catch (Exception e) {
			System.out.println(e);
			System.exit(0);
		}
		return object;
	}

	public static void writeObjectToFile(Object object, String filename) {
		try {
			OutputStream outputStream = new BufferedOutputStream(new FileOutputStream(filename));
			ObjectOutput objectOutput = new ObjectOutputStream(outputStream);
			try {
				objectOutput.writeObject(object);
			} finally {
				objectOutput.close();
			}
		} catch (Exception e) {
			System.out.println(e);
			System.exit(0);
		}
	}

	public static List<Integer> shuffleInstances(int end) {
		List<Integer> list = new ArrayList<Integer>();
		for (int i = 0; i < end; i++) {
			list.add(i);
		}
		Collections.shuffle(list);
		return list;
	}

	public static double getAverage(double[] values) {
		double sum = 0.0;
		for (int i = 0; i < values.length; i++) {
			sum += values[i];
		}
		return sum / values.length;
	}

	public static double logisticFunction(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}
}
