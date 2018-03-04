import os


def duration(start, end):
    duration_ = end - start
    minutes = int(duration_//60)
    seconds = int(duration_ % 60)
    if minutes > 60:
        hours = int(minutes//60)
        minutes = minutes % 60
    else:
        hours = 00
    time_str = "{}hh:{}mm:{}ss".format(hours, minutes, seconds)
    return time_str


def iterate_mini_batchOne(train, batch_size):

    indices = []
    values = []

    currCount = 0

    for _, val in train.items():
        if currCount >= batch_size:
            yield {'values': values, 'indices': indices}
            # restart the count
            currCount = 0
            indices = []
            values = []

        def add_index_counter(x): return [currCount, x]

        tmp_indices = list(map(add_index_counter, val['items']))
        indices += tmp_indices
        values += val['ratings']

        currCount += 1


def iterate_mini_batchTwo(test, train, batch_size):

    test_indices = []
    test_values = []

    train_indices = []
    train_values = []

    currCount = 0

    for key, val in test.items():
        if currCount >= batch_size:
            yield {'indices': test_indices, 'values': test_values}, {'indices': train_indices, 'values': train_values}

            currCount = 0
            test_indices = []
            test_values = []

            train_indices = []
            train_values = []

        if key in train:
            def add_index_counter(x): return [currCount, x]

            tmp_train_indices = list(
                map(add_index_counter, train[key]['items']))
            train_indices += tmp_train_indices
            train_values += train[key]['ratings']

            tmp_test_indices = list(map(add_index_counter, val['items']))
            test_indices += tmp_test_indices
            test_values += val['ratings']

            currCount += 1


def normalize_data(train, test, normal=0):

    if normal == 0:
        def reduce(x): return x/5.0
    else:
        def reduce(x): return (x-3)/2.0

    for _, val in train.items():
        val['ratings'] = list(map(reduce, val['ratings']))

    for _, val in test.items():
        val['ratings'] = list(map(reduce, val['ratings']))

    return train, test
