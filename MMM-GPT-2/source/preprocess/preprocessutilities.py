
def events_to_events_data(events):

    events = sorted(events, key=lambda event: event[2])

    events_data = []
    for event_index, event, event_next in zip(range(len(events)), events, events[1:] + [None]):
        if event_index == 0 and event[2] != 0.0:
            event_data = {
                "type": "TIME_DELTA",
                "delta": event[2]
            }
            events_data += [event_data]

        event_data = {
            "type": event[0],
            "pitch": event[1]
        }
        events_data += [event_data]

        if event_next is None:
            continue

        delta = event_next[2] - event[2]
        assert delta >= 0, events
        if delta != 0.0:
            event_data = {
                "type": "TIME_DELTA",
                "delta": delta
            }
            events_data += [event_data]

    return events_data
