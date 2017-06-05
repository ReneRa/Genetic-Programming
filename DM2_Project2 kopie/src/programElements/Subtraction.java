package programElements;

public class Subtraction extends Operator {
	
	private static final long serialVersionUID = 7L;

	public Subtraction() {
		super(2);
	}

	public double performOperation(double... arguments) {
		return arguments[0] - arguments[1];
	}
	
	public String toString() {
		return "-";
	}
}
