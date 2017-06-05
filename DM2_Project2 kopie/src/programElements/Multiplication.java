package programElements;

public class Multiplication extends Operator {

	private static final long serialVersionUID = 7L;

	public Multiplication() {
		super(2);
	}

	public double performOperation(double... arguments) {
		return arguments[0] * arguments[1];
	}

	public String toString() {
		return "*";
	}
}
