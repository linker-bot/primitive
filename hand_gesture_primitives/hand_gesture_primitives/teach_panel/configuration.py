"""Resolve side/model/topic identity before creating any ROS publisher or client."""
from ..hand_config import resolve_hand_config_key
from ..topic_defaults import resolve_side_topic, side_hand_topics


def resolve_hands(hand_side, hand_type, configured=(), overrides=None):
    side = str(hand_side).strip().lower()
    if side not in {'left', 'right'}:
        raise ValueError('hand_side must be left or right')
    model = resolve_hand_config_key(hand_type)
    requested = [str(value).strip() for value in (configured or ()) if str(value).strip()]
    models = {}
    for value in requested or [f'{side}:{model}']:
        parts = value.split(':')
        if len(parts) != 2 or parts[0] not in {'left', 'right'} or parts[0] in models:
            raise ValueError('workbench_hands must contain unique left:MODEL/right:MODEL values')
        models[parts[0]] = resolve_hand_config_key(parts[1])
    primary = side if side in models else next(iter(models))
    topics = side_hand_topics(primary)
    for key, value in (overrides or {}).items():
        topics[key] = resolve_side_topic(value, topics[key])
    return primary, models, topics
