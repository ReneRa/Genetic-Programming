package programElements;

public class Median extends Operator{
private static final long serialVersionUID = 7L;
	
	public Median(){
		super (2);
	}
	
	public double performOperation(double... arguments) {
				return (arguments[0] + arguments[1]) / 2;
	}
	
	public String toString() {
		return "med";
	}

}
