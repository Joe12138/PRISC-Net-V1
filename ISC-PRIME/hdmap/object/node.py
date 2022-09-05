class Node(object):
    def __init__(self, node_id: int, x: float, y: float, visible: bool = True):
        """
        Initialize the node object.
        :param node_id: The id of the node object.
        :param x: The x coordinate of the node object.
        :param y: The y coordinate of the node object.
        :param visible: If the node is visible?
        """
        self.id = node_id
        self.x = x
        self.y = y
        self.visible = visible

        # The ways which the node object belong to -> list of way_id
        self.way_list = []