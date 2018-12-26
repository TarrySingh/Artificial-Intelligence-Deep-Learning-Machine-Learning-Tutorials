package org.antlr.gunit.swingui.model;

public class TestCase {

    private ITestCaseInput input;
    private ITestCaseOutput output;
    private boolean pass;

    public boolean isPass() {
        return pass;
    }

    public void setPass(boolean value) {
        pass = value;
    }

    public ITestCaseInput getInput() {
        return this.input;
    }

    public ITestCaseOutput getOutput() {
        return this.output;
    }

    public TestCase(ITestCaseInput input, ITestCaseOutput output) {
        this.input = input;
        this.output = output;
    }

    @Override
    public String toString() {
        return String.format("[%s]->[%s]", input.getScript(), output.getScript());
    }

    public void setInput(ITestCaseInput in) {
        this.input = in;
    }

    public void setOutput(ITestCaseOutput out) {
        this.output = out;
    }

}
