class GraphNode(object):
    def __init__(self, lane_id: int):
        self.lane_id = lane_id
        self.parent = set()
        self.child = set()

    def add_child(self, child_obj, lane_change: bool):
        if not isinstance(child_obj, GraphNode):
            raise ValueError("GraphNoe has to add child which is GraphNode.")

        self.child.add((child_obj, lane_change))
        child_obj.parent.add((self, lane_change))

        return child_obj

    def add_parent(self, parent_obj, lane_change: bool):
        if not isinstance(parent_obj, GraphNode):
            raise ValueError("GraphNoe has to add parent which is GraphNode.")

        self.parent.add((parent_obj, lane_change))
        parent_obj.child.add((self, lane_change))

        return parent_obj

    def get_in_degree(self):
        """
        Get the in degree of current graph node.
        :return: The in degree of current graph node.
        """
        return len(self.parent)

    def get_out_degree(self):
        """
        Get the out degree of current graph node.
        :return: The out degree of current graph node.
        """
        return len(self.child)

    def get_in_degree_without_lane_change(self):
        """
        Get the in degree of current graph node without lane change.
        :return: The in degree of current graph node without lane change.
        """
        num = 0
        for parent_obj, lane_change in self.parent:
            if not lane_change:
                num += 1
        return num

    def get_out_degree_without_lane_change(self):
        """
        Get the out degree of current graph node without lane change.
        :return: The out degree of current graph node without lane change.
        """
        num = 0
        for child_obj, lane_change in self.child:
            if not lane_change:
                num += 1
        return num