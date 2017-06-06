package programElements;

public class Max extends Operator{
	private static final long serialVersionUID = 7L;
	
	public Max(){
		super (2);
	}
	
	public double performOperation(double... arguments) {
				return Math.max(arguments[0], arguments[1]);
	}
	
	public String toString() {
		return "max";
	}
}
