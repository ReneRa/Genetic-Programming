package programElements;

public class Exp extends Operator {

private static final long serialVersionUID = 7L;
	
	public Exp(){
		super (1);
	}
	
	public double performOperation(double... arguments) {
		return Math.exp(arguments[0]);
	}	
	
	public String toString() {
		return "exp";
	}
	
}
