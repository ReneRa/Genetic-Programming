package programElements;

public class Constant extends Terminal {

	private static final long serialVersionUID = 7L;

	protected double value;

	public Constant(Double value) {
		this.value = value;
	}

	public double getValue() {
		return value;
	}
	
	public void setValue(Double newValue) {
		this.value = newValue;
	}

	public String toString() {
		return "C" + value;
	}
}
