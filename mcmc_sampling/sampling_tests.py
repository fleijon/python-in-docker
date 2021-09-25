from sampling import plot_trace
from sampling import sample

def should_be_able_to_sample_and_create_traceplot():
    trace = sample()
    #plot_trace(trace)


if __name__ == '__main__':
    should_be_able_to_sample_and_create_traceplot()
