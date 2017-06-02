package programElements;

public class Expm1 extends Operator {

	private static final long serialVersionUID = 7L;
	
	public Expm1(){
		super (1);
	}
	
	public double performOperation(double... arguments) {
				return Math.expm1(arguments[0]);
		//		return Utils.aPowerB(arguments[0], arguments[1]);
	}
	
	public String toString() {
		return "expm1";
	}
}
	
