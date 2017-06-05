package programElements;

public abstract class Operator extends ProgramElement {
	
	private static final long serialVersionUID = 7L;

	public static final int DEFAULT_ARITY = 2;
	protected int arity;
	
	public Operator(int arity) {
		super();
		this.arity = arity;
	}

	public int getArity() {
		return arity;
	}
	
	public abstract double performOperation(double... arguments);
}
