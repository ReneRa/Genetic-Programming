package programElements;

public class ProtectedDivision extends Operator {

	private static final long serialVersionUID = 7L;

	public static final double MINIMUM_DENOMINATOR_VALUE = 0.00000000001;

	public ProtectedDivision() {
		super(2);
	}

	public double performOperation(double... arguments) {
		if (Math.abs(arguments[1]) < MINIMUM_DENOMINATOR_VALUE) {
			return arguments[0];
		} else {
			return arguments[0] / arguments[1];
		}
	}

	public String toString() {
		return "/";
	}
}
