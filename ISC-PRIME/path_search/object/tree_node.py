class TreeNode:
    def __init__(self, lane_id: int, lane_len: float, parent=None):
        self.lane_id = lane_id
        self.parent = parent
        self.child = {}
        self.cur_len = lane_len if self.parent is None else self.parent.cur_len+lane_len

        # For forward search: if self is leaf node -> remain_dist = dist between last_p and start_p in leaf lanelet
        # For backward search: if self is leaf node -> remain_dist = dist between last_p and end_p in leaf lanelet
        self.remain_dist = None

    def add_child(self, lane_id: int, lane_len: float, child_obj=None):
        """
        Add child for the self node.
        :param lane_id: The id of the child lane.
        :param lane_len: The length of the child lane.
        :param child_obj: The object of the child.
        :return: child object.
        """
        if child_obj and not isinstance(child_obj, TreeNode):
            raise ValueError("TreeNode only add another TreeNode object as child!")

        if child_obj is None:
            child_obj = TreeNode(lane_id=lane_id, lane_len=lane_len, parent=self)
        self.child[lane_id] = child_obj

        return child_obj

    def __repr__(self) -> str:
        return "TreeNode(%s)" % self.lane_id

    def items(self):
        return self.child.items()

    def dump(self, indent=0):
        """
        Dump tree to string
        :param indent:
        :return:
        """
        tab = '    ' * (indent - 1) + ' |- ' if indent > 0 else ''
        print('%s%s' % (tab, self.lane_id))
        for name, obj in self.items():
            obj.dump(indent + 1)
