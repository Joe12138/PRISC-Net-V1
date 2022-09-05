from typing import List


class Way(object):
    def __init__(self, way_id: int, way_type: str, way_subtype: str, ref_node_list: List[int]):
        """
        The initialization of the way object.
        :param way_id: The id of the way object.
        :param way_type: The type of the way object.
        :param way_subtype: The subtype of the way object.
        :param ref_node_list: The list of node's id which the way object contains.
        """
        self.id = way_id
        self.type = way_type
        self.subtype = way_subtype
        self.ref_node_list = ref_node_list

        # The id of the lanes which the way object belong to.
        self.lane_list = []